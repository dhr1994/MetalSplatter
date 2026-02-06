import Metal
import simd
import Synchronization
import os

/**
 SplatSorterGPU provides GPU-accelerated radix sort for splat depth ordering.

 This is a drop-in replacement for SplatSorter that uses Metal compute shaders
 to perform the depth sort on the GPU. For 200K splats, this reduces sort time
 from ~100-200ms (CPU) to ~5-10ms (GPU), eliminating the visual "pop" during rotation.

 ## Algorithm

 The implementation uses a 4-pass radix sort on 32-bit depth keys (8 bits per pass):
 1. Compute depth keys: Transform splat positions to sortable uint depths
 2. For each of 4 passes (bits 0-7, 8-15, 16-23, 24-31):
    a. Histogram: Count digit occurrences per threadgroup
    b. Prefix Sum: Compute exclusive scan for output offsets
    c. Scatter: Reorder elements to their sorted positions
 3. Convert: Transform sorted global indices back to ChunkedSplatIndex

 ## Buffer Management

 Like SplatSorter, this class maintains triple-buffered output for lock-free rendering.
 Unlike the CPU version, sorting happens synchronously on the GPU and completes quickly
 enough that we don't need the complex async polling mechanism.
 */
class SplatSorterGPU: @unchecked Sendable {

    // MARK: - Constants

    private static let bufferCount = 3
    private static let radixBits: UInt32 = 8
    private static let radixSize: UInt32 = 256
    private static let threadgroupSize: Int = 256
    private static let pollIntervalNanoseconds: UInt64 = 1_000_000 // 1ms

    // MARK: - Types

    /// Represents a chunk for sorting purposes (same as SplatSorter)
    struct ChunkReference {
        let chunkIndex: UInt16
        let buffer: MetalBuffer<EncodedSplatPoint>
    }

    // Keep in sync with ShaderCommon.h : SplatDepthKey
    private struct SplatDepthKey {
        var sortKey: UInt32
        var globalIndex: UInt32
    }

    // Keep in sync with ShaderCommon.h : GPUSortUniforms
    private struct GPUSortUniforms {
        var cameraPosition: MTLPackedFloat3
        var totalSplatCount: UInt32
        var bitOffset: UInt32
        var blockCount: UInt32
        var sortByDistance: UInt32
        var cameraForward: MTLPackedFloat3
        var _padding: UInt32 = 0
    }

    // Keep in sync with ShaderCommon.h : ChunkOffsetEntry
    private struct ChunkOffsetEntry {
        var startIndex: UInt32
        var chunkIndex: UInt16
        var _padding: UInt16 = 0
    }

    // Keep in sync with ShaderCommon.h : ChunkTable
    private struct GPUChunkTableHeader {
        var chunksPointer: UInt64
        var enabledChunkCount: UInt16
        var _padding: UInt16
        var _padding2: UInt32
    }

    // Keep in sync with ShaderCommon.h : ChunkInfo
    private struct GPUChunkInfo {
        var splatsPointer: UInt64
        var shCoefficientsPointer: UInt64
        var splatCount: UInt32
        var shDegree: UInt8
        var _shPadding: (UInt8, UInt8, UInt8) = (0, 0, 0)
    }

    private struct IndexBuffer {
        let buffer: MetalBuffer<ChunkedSplatIndex>
        var referenceCount: Int = 0
        var isValid: Bool = false
    }

    struct CameraPose: Equatable {
        var position: SIMD3<Float>
        var forward: SIMD3<Float>
    }

    private struct State {
        var indexBuffers: [IndexBuffer]
        var sortingBufferIndex: Int? = nil
        var mostRecentValidBufferIndex: Int? = nil
        var hasExclusiveAccess: Bool = false
        var pendingInvalidation: Bool = false
        var cameraPose: CameraPose? = nil
        var needsSort: Bool = false
        var chunks: [ChunkReference] = []
        var isReadingChunks: Bool = false
        var sortLoopRunning: Bool = false
    }

    // MARK: - Properties

    private let state: Mutex<State>
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private static let log = Logger(
        subsystem: Bundle.module.bundleIdentifier ?? "MetalSplatter",
        category: "SplatSorterGPU"
    )

    /// Called when a sort starts (for diagnostics)
    var onSortStart: (@Sendable () -> Void)?
    /// Called when a sort completes with duration
    var onSortComplete: (@Sendable (TimeInterval) -> Void)?

    // MARK: - Compute Pipeline States

    private let computeDepthsPipeline: MTLComputePipelineState
    private let radixHistogramPipeline: MTLComputePipelineState
    private let prefixSumLocalPipeline: MTLComputePipelineState
    private let addBlockSumsPipeline: MTLComputePipelineState
    private let radixScatterPipeline: MTLComputePipelineState
    private let convertToChunkedIndicesPipeline: MTLComputePipelineState

    // MARK: - Reusable Buffers

    // Double-buffered depth keys for ping-pong during radix passes
    private var depthKeysA: MetalBuffer<SplatDepthKey>?
    private var depthKeysB: MetalBuffer<SplatDepthKey>?

    // Histogram and prefix sum buffers
    // Layout: histogram[digit * blockCount + blockIndex]
    private var histogramBuffer: MTLBuffer?
    private var blockSumsBuffer: MTLBuffer?

    // Chunk offset table for final conversion
    private var chunkOffsetsBuffer: MetalBuffer<ChunkOffsetEntry>?

    // Chunk table buffer (rebuilt each sort if chunks change)
    private var chunkTableBuffer: MTLBuffer?

    // Track last known configuration to detect when buffers need resizing
    private var lastSplatCount: Int = 0
    private var lastBlockCount: Int = 0

    // MARK: - Initialization

    init(device: MTLDevice) throws {
        Self.log.info("Initializing SplatSorterGPU...")
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw NSError(domain: "SplatSorterGPU", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = commandQueue
        Self.log.info("Command queue created")

        // Load shader library
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
            Self.log.info("Shader library loaded")
        } catch {
            throw NSError(domain: "SplatSorterGPU", code: -2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to load shader library: \(error)"])
        }

        // Create compute pipelines
        func makePipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw NSError(domain: "SplatSorterGPU", code: -3,
                             userInfo: [NSLocalizedDescriptionKey: "Failed to find function: \(name)"])
            }
            return try device.makeComputePipelineState(function: function)
        }

        self.computeDepthsPipeline = try makePipeline("computeDepths")
        self.radixHistogramPipeline = try makePipeline("radixHistogram")
        self.prefixSumLocalPipeline = try makePipeline("prefixSumLocal")
        self.addBlockSumsPipeline = try makePipeline("addBlockSums")
        self.radixScatterPipeline = try makePipeline("radixScatter")
        self.convertToChunkedIndicesPipeline = try makePipeline("convertToChunkedIndices")
        Self.log.info("All 6 compute pipelines created")

        // Initialize index buffers
        var indexBuffers: [IndexBuffer] = []
        for _ in 0..<Self.bufferCount {
            let buffer = try MetalBuffer<ChunkedSplatIndex>(device: device)
            indexBuffers.append(IndexBuffer(buffer: buffer))
        }

        self.state = Mutex(State(indexBuffers: indexBuffers))
        Self.log.info("SplatSorterGPU initialization complete")
    }

    // MARK: - Chunk Management

    /// The current chunks being sorted
    var chunks: [ChunkReference] {
        get { state.withLock { $0.chunks } }
    }

    /// Total splat count across all chunks
    var totalSplatCount: Int {
        state.withLock { state in
            state.chunks.reduce(0) { $0 + $1.buffer.count }
        }
    }

    /// Sets the chunks to sort
    func setChunks(_ chunks: [ChunkReference]) {
        state.withLock { state in
            state.chunks = chunks
            state.needsSort = !chunks.isEmpty
        }
        ensureSortLoopRunning()
    }

    // MARK: - Camera Pose Updates

    /// Updates the camera pose, triggering a new sort if needed
    func updateCameraPose(position: SIMD3<Float>, forward: SIMD3<Float>) {
        state.withLock { state in
            state.cameraPose = CameraPose(position: position, forward: forward)
            state.needsSort = true
        }
        ensureSortLoopRunning()
    }

    // MARK: - Index Buffer Access (Scoped)

    /// Provides scoped access to sorted index buffer
    func withSortedIndices(_ body: (MetalBuffer<ChunkedSplatIndex>) throws -> Void) async rethrows {
        guard let buffer = await obtainSortedIndices() else { return }
        defer { releaseSortedIndices(buffer) }
        try body(buffer)
    }

    // MARK: - Index Buffer Access (Explicit)

    /// Obtains a reference to the current sorted index buffer
    func obtainSortedIndices() async -> MetalBuffer<ChunkedSplatIndex>? {
        while !Task.isCancelled {
            if let buffer = tryObtainSortedIndices() {
                return buffer
            }
            try? await Task.sleep(nanoseconds: Self.pollIntervalNanoseconds)
        }
        return nil
    }

    /// Attempts to obtain sorted indices without waiting
    func tryObtainSortedIndices() -> MetalBuffer<ChunkedSplatIndex>? {
        state.withLock { state -> MetalBuffer<ChunkedSplatIndex>? in
            guard !state.hasExclusiveAccess else { return nil }

            guard let validIndex = state.mostRecentValidBufferIndex,
                  state.indexBuffers[validIndex].isValid else {
                return nil
            }

            state.indexBuffers[validIndex].referenceCount += 1
            return state.indexBuffers[validIndex].buffer
        }
    }

    /// Releases a previously obtained index buffer reference
    func releaseSortedIndices(_ buffer: MetalBuffer<ChunkedSplatIndex>) {
        state.withLock { state in
            guard let index = state.indexBuffers.firstIndex(where: { $0.buffer === buffer }) else {
                assertionFailure("Released buffer not found in index buffers")
                return
            }
            assert(state.indexBuffers[index].referenceCount > 0, "Reference count underflow")
            state.indexBuffers[index].referenceCount -= 1
        }
    }

    /// Invalidates all index buffers synchronously
    func invalidateAllBuffers() {
        state.withLock { state in
            for i in 0..<state.indexBuffers.count {
                state.indexBuffers[i].isValid = false
            }
            state.mostRecentValidBufferIndex = nil
            state.needsSort = true
        }
    }

    // MARK: - Exclusive Access

    /// Provides exclusive access to update chunks
    func withExclusiveAccess(invalidateIndexBuffers: Bool = true,
                             _ body: () async throws -> Void) async rethrows {
        // Wait until not reading chunks
        while !Task.isCancelled {
            let canProceed = state.withLock { state -> Bool in
                if state.isReadingChunks {
                    return false
                }
                state.hasExclusiveAccess = true
                if invalidateIndexBuffers {
                    state.pendingInvalidation = true
                }
                return true
            }

            if canProceed {
                break
            }

            try? await Task.sleep(nanoseconds: Self.pollIntervalNanoseconds)
        }

        defer {
            state.withLock { state in
                state.hasExclusiveAccess = false
                state.pendingInvalidation = false
            }
        }

        // If invalidating, wait for all references to be released
        if invalidateIndexBuffers {
            while !Task.isCancelled {
                let allReleased = state.withLock { state -> Bool in
                    state.indexBuffers.allSatisfy { $0.referenceCount == 0 }
                }

                if allReleased {
                    state.withLock { state in
                        for i in 0..<state.indexBuffers.count {
                            state.indexBuffers[i].isValid = false
                        }
                        state.mostRecentValidBufferIndex = nil
                    }
                    break
                }

                try? await Task.sleep(nanoseconds: Self.pollIntervalNanoseconds)
            }
        }

        try await body()

        let shouldTriggerSort = state.withLock { state -> Bool in
            state.needsSort = !state.chunks.isEmpty
            return state.needsSort
        }

        if shouldTriggerSort {
            ensureSortLoopRunning()
        }
    }

    // MARK: - Sort Loop

    private func ensureSortLoopRunning() {
        let shouldStart = state.withLock { state -> Bool in
            if state.sortLoopRunning {
                return false
            }
            state.sortLoopRunning = true
            return true
        }

        if shouldStart {
            Task.detached(priority: .high) { [weak self] in
                await self?.sortLoop()
            }
        }
    }

    private func sortLoop() async {
        defer {
            state.withLock { state in
                state.sortLoopRunning = false
            }
        }

        while !Task.isCancelled {
            let sortParams = state.withLock { state -> (chunks: [ChunkReference], pose: CameraPose, bufferIndex: Int)? in
                guard !state.hasExclusiveAccess else { return nil }

                guard state.needsSort,
                      !state.chunks.isEmpty,
                      let pose = state.cameraPose else {
                    return nil
                }

                guard let bufferIndex = state.indexBuffers.firstIndex(where: { $0.referenceCount == 0 }) else {
                    return nil
                }

                state.sortingBufferIndex = bufferIndex
                state.isReadingChunks = true
                state.needsSort = false

                return (state.chunks, pose, bufferIndex)
            }

            guard let params = sortParams else {
                let shouldExit = state.withLock { state -> Bool in
                    !state.needsSort && state.chunks.isEmpty
                }

                if shouldExit {
                    return
                }

                try? await Task.sleep(nanoseconds: Self.pollIntervalNanoseconds)
                continue
            }

            // Perform the GPU sort
            performGPUSort(
                chunks: params.chunks,
                cameraPose: params.pose,
                targetBufferIndex: params.bufferIndex
            )
        }
    }

    // MARK: - GPU Sort Implementation

    private func performGPUSort(
        chunks: [ChunkReference],
        cameraPose: CameraPose,
        targetBufferIndex: Int
    ) {
        let startTime = Date()
        onSortStart?()

        let totalSplatCount = chunks.reduce(0) { $0 + $1.buffer.count }
        let targetBuffer = state.withLock { $0.indexBuffers[targetBufferIndex].buffer }

        // Mark that we're done reading chunk metadata (we'll access GPU buffers directly)
        state.withLock { state in
            state.isReadingChunks = false
        }

        guard totalSplatCount > 0 else {
            state.withLock { state in
                state.sortingBufferIndex = nil
            }
            return
        }

        // Ensure output buffer has capacity
        do {
            try targetBuffer.ensureCapacity(totalSplatCount)
            targetBuffer.count = totalSplatCount
        } catch {
            Self.log.error("Failed to resize output buffer: \(error)")
            state.withLock { state in
                state.sortingBufferIndex = nil
            }
            return
        }

        // Ensure working buffers are sized correctly
        do {
            try ensureBufferCapacity(splatCount: totalSplatCount)
        } catch {
            Self.log.error("Failed to resize working buffers: \(error)")
            state.withLock { state in
                state.sortingBufferIndex = nil
            }
            return
        }

        // Build chunk table for GPU access
        guard let chunkTableBuffer = buildChunkTableBuffer(chunks: chunks) else {
            Self.log.error("Failed to build chunk table buffer")
            state.withLock { state in
                state.sortingBufferIndex = nil
            }
            return
        }
        self.chunkTableBuffer = chunkTableBuffer

        // Build chunk offsets for final conversion
        buildChunkOffsetsBuffer(chunks: chunks)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Self.log.error("Failed to create command buffer")
            state.withLock { state in
                state.sortingBufferIndex = nil
            }
            return
        }
        commandBuffer.label = "GPU Sort"

        let blockCount = (totalSplatCount + Self.threadgroupSize - 1) / Self.threadgroupSize

        // Prepare uniforms
        var uniforms = GPUSortUniforms(
            cameraPosition: MTLPackedFloat3Make(cameraPose.position.x, cameraPose.position.y, cameraPose.position.z),
            totalSplatCount: UInt32(totalSplatCount),
            bitOffset: 0,
            blockCount: UInt32(blockCount),
            sortByDistance: SplatRenderer.Constants.sortByDistance ? 1 : 0,
            cameraForward: MTLPackedFloat3Make(cameraPose.forward.x, cameraPose.forward.y, cameraPose.forward.z)
        )

        guard let depthKeysA = depthKeysA,
              let depthKeysB = depthKeysB,
              let histogramBuffer = histogramBuffer,
              let blockSumsBuffer = blockSumsBuffer,
              let chunkOffsetsBuffer = chunkOffsetsBuffer else {
            Self.log.error("Working buffers not initialized")
            state.withLock { state in
                state.sortingBufferIndex = nil
            }
            return
        }

        // ========================================
        // Pass 1: Compute Depth Keys
        // ========================================
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Compute Depths"
            encoder.setComputePipelineState(computeDepthsPipeline)
            encoder.setBuffer(chunkTableBuffer, offset: 0, index: 0)
            encoder.setBuffer(depthKeysA.buffer, offset: 0, index: 1)
            encoder.setBytes(&uniforms, length: MemoryLayout<GPUSortUniforms>.size, index: 2)

            // Make chunk splat buffers resident
            for chunk in chunks {
                encoder.useResource(chunk.buffer.buffer, usage: .read)
            }

            let threadsPerGrid = MTLSize(width: totalSplatCount, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        // ========================================
        // Passes 2-5: 4 Radix Sort Passes (8 bits each)
        // ========================================
        var inputBuffer = depthKeysA.buffer
        var outputBuffer = depthKeysB.buffer

        for pass in 0..<4 {
            uniforms.bitOffset = UInt32(pass * 8)

            // Clear histogram
            if let encoder = commandBuffer.makeBlitCommandEncoder() {
                encoder.label = "Clear Histogram Pass \(pass)"
                encoder.fill(buffer: histogramBuffer, range: 0..<histogramBuffer.length, value: 0)
                encoder.endEncoding()
            }

            // Histogram
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Histogram Pass \(pass)"
                encoder.setComputePipelineState(radixHistogramPipeline)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(histogramBuffer, offset: 0, index: 1)
                encoder.setBytes(&uniforms, length: MemoryLayout<GPUSortUniforms>.size, index: 2)

                let threadsPerGrid = MTLSize(width: totalSplatCount, height: 1, depth: 1)
                let threadsPerThreadgroup = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
                encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }

            // Prefix Sum - single threadgroup of 256 threads
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Prefix Sum Pass \(pass)"
                encoder.setComputePipelineState(prefixSumLocalPipeline)
                encoder.setBuffer(histogramBuffer, offset: 0, index: 0)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
                encoder.setBytes(&uniforms, length: MemoryLayout<GPUSortUniforms>.size, index: 2)

                // Single threadgroup with 256 threads for 256-element prefix sum
                let threadsPerThreadgroup = MTLSize(width: Int(Self.radixSize), height: 1, depth: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }

            // Note: addBlockSums pass removed - not needed in simplified approach

            // Scatter
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Scatter Pass \(pass)"
                encoder.setComputePipelineState(radixScatterPipeline)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(outputBuffer, offset: 0, index: 1)
                encoder.setBuffer(histogramBuffer, offset: 0, index: 2)
                encoder.setBytes(&uniforms, length: MemoryLayout<GPUSortUniforms>.size, index: 3)

                let threadsPerGrid = MTLSize(width: totalSplatCount, height: 1, depth: 1)
                let threadsPerThreadgroup = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
                encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }

            // Swap buffers for next pass
            swap(&inputBuffer, &outputBuffer)
        }

        // After 4 passes, sorted data is in inputBuffer (due to final swap)

        // ========================================
        // Pass 6: Convert to Chunked Indices
        // ========================================
        var splatCount = UInt32(totalSplatCount)
        var chunkCount = UInt32(chunks.count)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Convert to Chunked Indices"
            encoder.setComputePipelineState(convertToChunkedIndicesPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(chunkOffsetsBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(targetBuffer.buffer, offset: 0, index: 2)
            encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&chunkCount, length: MemoryLayout<UInt32>.size, index: 4)

            let threadsPerGrid = MTLSize(width: totalSplatCount, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Mark buffer as valid
        let wasInvalidated = state.withLock { state -> Bool in
            state.sortingBufferIndex = nil

            if state.pendingInvalidation {
                return true
            }

            state.indexBuffers[targetBufferIndex].isValid = true
            state.mostRecentValidBufferIndex = targetBufferIndex
            return false
        }

        if !wasInvalidated {
            let duration = -startTime.timeIntervalSinceNow
            Self.log.info("GPU sort completed: \(totalSplatCount) splats in \(String(format: "%.1f", duration * 1000))ms")
            onSortComplete?(duration)
        }
    }

    // MARK: - Buffer Management

    private func ensureBufferCapacity(splatCount: Int) throws {
        let blockCount = (splatCount + Self.threadgroupSize - 1) / Self.threadgroupSize

        // Depth key buffers (double-buffered)
        if depthKeysA == nil || depthKeysA!.capacity < splatCount {
            depthKeysA = try MetalBuffer<SplatDepthKey>(device: device, capacity: splatCount)
            depthKeysA!.buffer.label = "Depth Keys A"
        }
        if depthKeysB == nil || depthKeysB!.capacity < splatCount {
            depthKeysB = try MetalBuffer<SplatDepthKey>(device: device, capacity: splatCount)
            depthKeysB!.buffer.label = "Depth Keys B"
        }

        // Histogram buffer: just 256 elements (simplified approach)
        let histogramSize = Int(Self.radixSize) * MemoryLayout<UInt32>.stride
        if histogramBuffer == nil || histogramBuffer!.length < histogramSize {
            histogramBuffer = device.makeBuffer(length: histogramSize, options: .storageModeShared)
            histogramBuffer?.label = "Histogram"
        }

        // Block sums buffer: 256 digits (kept for API compatibility, not used)
        let blockSumsSize = Int(Self.radixSize) * MemoryLayout<UInt32>.stride
        if blockSumsBuffer == nil || blockSumsBuffer!.length < blockSumsSize {
            blockSumsBuffer = device.makeBuffer(length: blockSumsSize, options: .storageModeShared)
            blockSumsBuffer?.label = "Block Sums"
        }

        lastSplatCount = splatCount
        lastBlockCount = blockCount
    }

    private func buildChunkTableBuffer(chunks: [ChunkReference]) -> MTLBuffer? {
        let headerSize = MemoryLayout<GPUChunkTableHeader>.size
        let chunkInfoSize = MemoryLayout<GPUChunkInfo>.stride
        let chunksArraySize = chunks.count * chunkInfoSize
        let requiredSize = headerSize + chunksArraySize

        guard let buffer = device.makeBuffer(length: requiredSize, options: .storageModeShared) else {
            return nil
        }
        buffer.label = "GPU Sort Chunk Table"

        let ptr = buffer.contents()

        // Write header
        let headerPtr = ptr.assumingMemoryBound(to: GPUChunkTableHeader.self)
        headerPtr.pointee = GPUChunkTableHeader(
            chunksPointer: buffer.gpuAddress + UInt64(headerSize),
            enabledChunkCount: UInt16(chunks.count),
            _padding: 0,
            _padding2: 0
        )

        // Write chunk info array
        let chunksPtr = ptr.advanced(by: headerSize).assumingMemoryBound(to: GPUChunkInfo.self)
        for (index, chunk) in chunks.enumerated() {
            chunksPtr[index] = GPUChunkInfo(
                splatsPointer: chunk.buffer.buffer.gpuAddress,
                shCoefficientsPointer: 0,  // Not needed for sorting
                splatCount: UInt32(chunk.buffer.count),
                shDegree: 0
            )
        }

        return buffer
    }

    private func buildChunkOffsetsBuffer(chunks: [ChunkReference]) {
        if chunkOffsetsBuffer == nil || chunkOffsetsBuffer!.capacity < chunks.count {
            chunkOffsetsBuffer = try? MetalBuffer<ChunkOffsetEntry>(device: device, capacity: max(chunks.count, 1))
            chunkOffsetsBuffer?.buffer.label = "Chunk Offsets"
        }

        guard let buffer = chunkOffsetsBuffer else { return }

        buffer.count = chunks.count
        var runningCount: UInt32 = 0
        for (index, chunk) in chunks.enumerated() {
            buffer.values[index] = ChunkOffsetEntry(
                startIndex: runningCount,
                chunkIndex: chunk.chunkIndex
            )
            runningCount += UInt32(chunk.buffer.count)
        }
    }
}
