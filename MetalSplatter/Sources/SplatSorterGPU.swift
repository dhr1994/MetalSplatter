import Metal
import simd
import Synchronization
import os

/**
 SplatSorterGPU provides GPU-accelerated bitonic sort for splat depth ordering.

 This is a drop-in replacement for SplatSorter that uses Metal compute shaders
 to perform the depth sort on the GPU. For 200K splats, this reduces sort time
 from ~100-200ms (CPU) to ~2-5ms (GPU), eliminating the visual "pop" during rotation.

 ## Algorithm

 The implementation uses a bitonic sort on 32-bit depth keys:
 1. Compute depth keys: Transform splat positions to sortable uint depths
 2. Pad to next power of 2 with UINT_MAX sentinel values
 3. bitonicSortLocal: Sort 512-element blocks in shared memory (stages 0-8)
 4. For stages 9..log2(N)-1:
    a. bitonicMergeGlobal: Compare-and-swap for large strides (≥512)
    b. bitonicMergeLocal: Finish stage in shared memory (strides <512)
 5. Convert: Transform sorted global indices back to ChunkedSplatIndex

 Bitonic sort is deterministic (no atomics) and naturally suited to GPU parallelism.
 It sorts in-place, needing only a single depth key buffer.

 ## Buffer Management

 Like SplatSorter, this class maintains triple-buffered output for lock-free rendering.
 Sorting happens synchronously on the GPU and completes quickly enough that we don't
 need the complex async polling mechanism.
 */
class SplatSorterGPU: @unchecked Sendable {

    // MARK: - Constants

    private static let bufferCount = 3
    private static let threadgroupSize: Int = 256
    private static let blockSize: Int = 512       // Elements per shared-memory block
    private static let log2Block: Int = 9          // log2(512)
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
        var bitonicStage: UInt32
        var bitonicStep: UInt32
        var sortByDistance: UInt32
        var cameraForward: MTLPackedFloat3
        var paddedCount: UInt32
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
    private let bitonicSortLocalPipeline: MTLComputePipelineState
    private let bitonicMergeGlobalPipeline: MTLComputePipelineState
    private let bitonicMergeLocalPipeline: MTLComputePipelineState
    private let convertToChunkedIndicesPipeline: MTLComputePipelineState

    // MARK: - Reusable Buffers

    // Single depth key buffer (bitonic sort is in-place)
    private var depthKeys: MetalBuffer<SplatDepthKey>?

    // Chunk offset table for final conversion
    private var chunkOffsetsBuffer: MetalBuffer<ChunkOffsetEntry>?

    // Chunk table buffer (rebuilt each sort if chunks change)
    private var chunkTableBuffer: MTLBuffer?

    // Track last known configuration to detect when buffers need resizing
    private var lastPaddedCount: Int = 0

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
        self.bitonicSortLocalPipeline = try makePipeline("bitonicSortLocal")
        self.bitonicMergeGlobalPipeline = try makePipeline("bitonicMergeGlobal")
        self.bitonicMergeLocalPipeline = try makePipeline("bitonicMergeLocal")
        self.convertToChunkedIndicesPipeline = try makePipeline("convertToChunkedIndices")
        Self.log.info("All 5 compute pipelines created (bitonic sort)")

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

        let paddedCount = nextPowerOfTwo(totalSplatCount)
        let log2N = Int(log2(Double(paddedCount)))

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
            try ensureBufferCapacity(paddedCount: paddedCount)
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
        commandBuffer.label = "GPU Bitonic Sort"

        guard let depthKeys = depthKeys,
              let chunkOffsetsBuffer = chunkOffsetsBuffer else {
            Self.log.error("Working buffers not initialized")
            state.withLock { state in
                state.sortingBufferIndex = nil
            }
            return
        }

        // Prepare uniforms
        var uniforms = GPUSortUniforms(
            cameraPosition: MTLPackedFloat3Make(cameraPose.position.x, cameraPose.position.y, cameraPose.position.z),
            totalSplatCount: UInt32(totalSplatCount),
            bitonicStage: 0,
            bitonicStep: 0,
            sortByDistance: SplatRenderer.Constants.sortByDistance ? 1 : 0,
            cameraForward: MTLPackedFloat3Make(cameraPose.forward.x, cameraPose.forward.y, cameraPose.forward.z),
            paddedCount: UInt32(paddedCount)
        )

        // ========================================
        // Pass 1: Compute Depth Keys
        // ========================================
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Compute Depths"
            encoder.setComputePipelineState(computeDepthsPipeline)
            encoder.setBuffer(chunkTableBuffer, offset: 0, index: 0)
            encoder.setBuffer(depthKeys.buffer, offset: 0, index: 1)
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
        // Pass 1.5: Fill padding with UINT_MAX
        // ========================================
        if paddedCount > totalSplatCount {
            if let encoder = commandBuffer.makeBlitCommandEncoder() {
                encoder.label = "Fill Padding"
                let startByte = totalSplatCount * MemoryLayout<SplatDepthKey>.stride
                let endByte = paddedCount * MemoryLayout<SplatDepthKey>.stride
                encoder.fill(buffer: depthKeys.buffer, range: startByte..<endByte, value: 0xFF)
                encoder.endEncoding()
            }
        }

        // ========================================
        // Pass 2: Bitonic Sort Local (stages 0-8)
        // ========================================
        let numBlocks = paddedCount / Self.blockSize
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Bitonic Sort Local"
            encoder.setComputePipelineState(bitonicSortLocalPipeline)
            encoder.setBuffer(depthKeys.buffer, offset: 0, index: 0)
            encoder.setBytes(&uniforms, length: MemoryLayout<GPUSortUniforms>.size, index: 1)

            let threadgroupCount = MTLSize(width: numBlocks, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }

        // ========================================
        // Pass 3: Bitonic Merge (stages 9..log2N-1)
        // ========================================
        for stage in Self.log2Block..<log2N {
            // Global merge steps: from step=stage down to step=log2Block (stride >= 512)
            for step in stride(from: stage, through: Self.log2Block, by: -1) {
                uniforms.bitonicStage = UInt32(stage)
                uniforms.bitonicStep = UInt32(step)

                if let encoder = commandBuffer.makeComputeCommandEncoder() {
                    encoder.label = "Bitonic Merge Global s\(stage) p\(step)"
                    encoder.setComputePipelineState(bitonicMergeGlobalPipeline)
                    encoder.setBuffer(depthKeys.buffer, offset: 0, index: 0)
                    encoder.setBytes(&uniforms, length: MemoryLayout<GPUSortUniforms>.size, index: 1)

                    let threadsPerGrid = MTLSize(width: paddedCount / 2, height: 1, depth: 1)
                    let threadsPerThreadgroup = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
                    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                    encoder.endEncoding()
                }
            }

            // Local merge: finish remaining steps (stride < 512) in shared memory
            uniforms.bitonicStage = UInt32(stage)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Bitonic Merge Local s\(stage)"
                encoder.setComputePipelineState(bitonicMergeLocalPipeline)
                encoder.setBuffer(depthKeys.buffer, offset: 0, index: 0)
                encoder.setBytes(&uniforms, length: MemoryLayout<GPUSortUniforms>.size, index: 1)

                let threadgroupCount = MTLSize(width: numBlocks, height: 1, depth: 1)
                let threadsPerThreadgroup = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }
        }

        // ========================================
        // Pass 4: Convert to Chunked Indices
        // ========================================
        var splatCount = UInt32(totalSplatCount)
        var chunkCount = UInt32(chunks.count)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Convert to Chunked Indices"
            encoder.setComputePipelineState(convertToChunkedIndicesPipeline)
            encoder.setBuffer(depthKeys.buffer, offset: 0, index: 0)
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
            // Removed verbose per-frame logging - was flooding logs at 60fps × 2 models = 120 lines/sec
            // Self.log.info("GPU bitonic sort completed: \(totalSplatCount) splats (padded to \(paddedCount)) in \(String(format: "%.1f", duration * 1000))ms")
            onSortComplete?(duration)
        }
    }

    // MARK: - Buffer Management

    private func ensureBufferCapacity(paddedCount: Int) throws {
        guard paddedCount != lastPaddedCount else { return }

        // Single depth key buffer (bitonic sort is in-place)
        if depthKeys == nil || depthKeys!.capacity < paddedCount {
            depthKeys = try MetalBuffer<SplatDepthKey>(device: device, capacity: paddedCount)
            depthKeys!.buffer.label = "Depth Keys"
            Self.log.info("Allocated depth keys buffer: \(paddedCount) elements (\(String(format: "%.1f", Double(paddedCount * MemoryLayout<SplatDepthKey>.stride) / 1_048_576))MB)")
        }

        lastPaddedCount = paddedCount
    }

    private func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 1 else { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }

    private func buildChunkTableBuffer(chunks: [ChunkReference]) -> MTLBuffer? {
        let headerSize = MemoryLayout<GPUChunkTableHeader>.size
        let chunkInfoSize = MemoryLayout<GPUChunkInfo>.stride
        let chunksArraySize = chunks.count * chunkInfoSize
        let requiredSize = headerSize + chunksArraySize

        // Pool the buffer - only allocate if we don't have one or it's too small
        if chunkTableBuffer == nil || chunkTableBuffer!.length < requiredSize {
            guard let buffer = device.makeBuffer(length: requiredSize, options: .storageModeShared) else {
                return nil
            }
            buffer.label = "GPU Sort Chunk Table"
            chunkTableBuffer = buffer
        }

        guard let buffer = chunkTableBuffer else { return nil }

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
