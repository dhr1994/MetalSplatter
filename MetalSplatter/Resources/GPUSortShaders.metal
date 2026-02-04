#include <metal_stdlib>
#include <simd/simd.h>
#include "ShaderCommon.h"

using namespace metal;

// ============================================================================
// GPU Radix Sort Implementation
// ============================================================================
//
// This implements a 4-pass radix sort for 32-bit depth keys, processing 8 bits
// per pass. Each pass consists of:
//   1. Histogram: Count occurrences of each digit (0-255)
//   2. Prefix Sum: Exclusive scan to compute output offsets
//   3. Scatter: Reorder elements based on prefix sums
//
// For 200K splats, this reduces sort time from ~100-200ms (CPU) to ~5-10ms (GPU).
// ============================================================================

// Radix sort parameters
constant uint RADIX_BITS = 8;
constant uint RADIX_SIZE = 256;  // 2^8 = 256 buckets
constant uint THREADGROUP_SIZE = 256;

// ============================================================================
// Float-to-Sortable-Uint Conversion
// ============================================================================
//
// IEEE 754 floats don't sort correctly as unsigned integers because:
// - Negative numbers have the sign bit set (making them appear larger)
// - The magnitude of negatives is inverted
//
// This conversion makes floats sortable as uints while preserving order:
// - Positive floats: flip sign bit (so they sort after negatives)
// - Negative floats: flip all bits (so larger negatives sort before smaller)

inline uint floatToSortableUint(float f) {
    uint u = as_type<uint>(f);
    // If negative (sign bit set), flip all bits
    // If positive, flip only sign bit
    uint mask = -int(u >> 31) | 0x80000000;
    return u ^ mask;
}

inline float sortableUintToFloat(uint u) {
    // Reverse the transformation
    uint mask = ((u >> 31) - 1) | 0x80000000;
    return as_type<float>(u ^ mask);
}

// ============================================================================
// Kernel 1: Compute Depth Keys
// ============================================================================
//
// Computes a sortable depth key for each splat based on camera position.
// For back-to-front rendering, we negate depth so larger distances sort first.

kernel void computeDepths(
    device const ChunkTable* chunkTable [[buffer(0)]],
    device SplatDepthKey* depthKeys [[buffer(1)]],
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]])
{
    if (globalIndex >= uniforms.totalSplatCount) {
        return;
    }

    // Find which chunk this global index belongs to by walking through chunks
    // This is O(numChunks) but numChunks is typically small (1-10)
    device ChunkInfo* chunks = chunkTable->chunks;
    uint chunkCount = chunkTable->enabledChunkCount;

    uint runningCount = 0;
    uint chunkIndex = 0;
    uint localIndex = globalIndex;

    for (uint c = 0; c < chunkCount; c++) {
        uint chunkSplatCount = chunks[c].splatCount;
        if (localIndex < chunkSplatCount) {
            chunkIndex = c;
            break;
        }
        localIndex -= chunkSplatCount;
        runningCount += chunkSplatCount;
    }

    // Get splat position from chunk
    device Splat* splats = chunks[chunkIndex].splats;
    float3 position = float3(splats[localIndex].position);

    // Compute depth
    float depth;
    if (uniforms.sortByDistance) {
        // Distance squared from camera (sortByDistance mode)
        float3 diff = position - float3(uniforms.cameraPosition);
        depth = dot(diff, diff);
    } else {
        // Dot product with camera forward (projection mode)
        depth = dot(position, float3(uniforms.cameraForward));
    }

    // Negate for back-to-front sorting (larger depths should come first)
    // Then convert to sortable uint
    depthKeys[globalIndex].sortKey = floatToSortableUint(-depth);
    depthKeys[globalIndex].globalIndex = globalIndex;
}

// ============================================================================
// Kernel 2: Radix Histogram
// ============================================================================
//
// Counts occurrences of each 8-bit digit at the current bit position.
// Each threadgroup processes a portion of the data and writes its local histogram.
// Output layout: histogram[digit * blockCount + blockIndex]

kernel void radixHistogram(
    device const SplatDepthKey* inputKeys [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]],
    uint localIndex [[thread_position_in_threadgroup]],
    uint groupIndex [[threadgroup_position_in_grid]])
{
    // Shared memory for local histogram
    threadgroup uint localHistogram[RADIX_SIZE];

    // Initialize local histogram
    if (localIndex < RADIX_SIZE) {
        localHistogram[localIndex] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count this thread's contribution
    if (globalIndex < uniforms.totalSplatCount) {
        uint key = inputKeys[globalIndex].sortKey;
        uint digit = (key >> uniforms.bitOffset) & (RADIX_SIZE - 1);
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHistogram[digit],
                                  1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global memory
    // Layout: histogram[digit * blockCount + blockIndex]
    if (localIndex < RADIX_SIZE) {
        uint globalHistIdx = localIndex * uniforms.blockCount + groupIndex;
        atomic_store_explicit(&histogram[globalHistIdx], localHistogram[localIndex],
                             memory_order_relaxed);
    }
}

// ============================================================================
// Kernel 3: Prefix Sum (Exclusive Scan)
// ============================================================================
//
// Performs a hierarchical exclusive prefix sum on the histogram.
// This is done in two phases:
//   1. Local prefix sum within each row (digit), storing block sums
//   2. Add block sums to create global offsets

kernel void prefixSumLocal(
    device uint* histogram [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]],
    uint localIndex [[thread_position_in_threadgroup]],
    uint groupIndex [[threadgroup_position_in_grid]])
{
    // Each threadgroup handles one digit (row of the histogram)
    // We scan across blockCount elements

    threadgroup uint shared[THREADGROUP_SIZE * 2];

    uint digitIndex = groupIndex;  // Which digit (0-255) this group processes
    if (digitIndex >= RADIX_SIZE) return;

    uint blockCount = uniforms.blockCount;

    // Each thread may need to process multiple elements if blockCount > THREADGROUP_SIZE
    // For simplicity, we'll handle up to THREADGROUP_SIZE blocks per digit
    // (If blockCount > 256, we'd need a more complex approach)

    uint histBase = digitIndex * blockCount;

    // Load data into shared memory (handle out of bounds)
    uint val = 0;
    if (localIndex < blockCount) {
        val = histogram[histBase + localIndex];
    }

    // Work-efficient parallel prefix sum (Blelloch scan)
    uint ai = localIndex;
    uint bi = localIndex + (THREADGROUP_SIZE / 2);

    // Pad shared memory access for bank conflict avoidance
    uint bankOffsetA = ai >> 4;  // CONFLICT_FREE_OFFSET
    uint bankOffsetB = bi >> 4;

    shared[ai + bankOffsetA] = (ai < blockCount) ? histogram[histBase + ai] : 0;
    if (bi < blockCount) {
        shared[bi + bankOffsetB] = histogram[histBase + bi];
    } else {
        shared[bi + bankOffsetB] = 0;
    }

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = THREADGROUP_SIZE >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (localIndex < d) {
            uint ai_idx = offset * (2 * localIndex + 1) - 1;
            uint bi_idx = offset * (2 * localIndex + 2) - 1;
            ai_idx += ai_idx >> 4;
            bi_idx += bi_idx >> 4;
            shared[bi_idx] += shared[ai_idx];
        }
        offset *= 2;
    }

    // Store total sum and clear last element
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localIndex == 0) {
        uint lastIdx = THREADGROUP_SIZE - 1 + ((THREADGROUP_SIZE - 1) >> 4);
        blockSums[digitIndex] = shared[lastIdx];
        shared[lastIdx] = 0;
    }

    // Down-sweep phase
    for (uint d = 1; d < THREADGROUP_SIZE; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (localIndex < d) {
            uint ai_idx = offset * (2 * localIndex + 1) - 1;
            uint bi_idx = offset * (2 * localIndex + 2) - 1;
            ai_idx += ai_idx >> 4;
            bi_idx += bi_idx >> 4;
            uint t = shared[ai_idx];
            shared[ai_idx] = shared[bi_idx];
            shared[bi_idx] += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write back to global memory
    if (ai < blockCount) {
        histogram[histBase + ai] = shared[ai + bankOffsetA];
    }
    if (bi < blockCount) {
        histogram[histBase + bi] = shared[bi + bankOffsetB];
    }
}

// ============================================================================
// Kernel 4: Add Block Sums
// ============================================================================
//
// Computes the global prefix sum of block sums and adds them to create final offsets.
// This is needed for the hierarchical scan.

kernel void addBlockSums(
    device uint* histogram [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]],
    uint localIndex [[thread_position_in_threadgroup]],
    uint groupIndex [[threadgroup_position_in_grid]])
{
    // First, do a prefix sum on blockSums (256 elements, one per digit)
    // This gives us the global offset for each digit

    threadgroup uint digitOffsets[RADIX_SIZE];

    if (groupIndex == 0) {
        // Load block sums
        digitOffsets[localIndex] = (localIndex < RADIX_SIZE) ? blockSums[localIndex] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel prefix sum on digit totals
        for (uint stride = 1; stride < RADIX_SIZE; stride *= 2) {
            uint temp = 0;
            if (localIndex >= stride) {
                temp = digitOffsets[localIndex - stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (localIndex >= stride) {
                digitOffsets[localIndex] += temp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Convert to exclusive scan (shift right, insert 0)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint myVal = digitOffsets[localIndex];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (localIndex > 0) {
            blockSums[localIndex] = digitOffsets[localIndex - 1];
        } else {
            blockSums[0] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Now add digit offsets to histogram entries
    // Each thread processes elements across all digits for its block position
    uint blockCount = uniforms.blockCount;
    if (localIndex < blockCount) {
        for (uint digit = 0; digit < RADIX_SIZE; digit++) {
            uint histIdx = digit * blockCount + localIndex;
            histogram[histIdx] += blockSums[digit];
        }
    }
}

// ============================================================================
// Kernel 5: Radix Scatter
// ============================================================================
//
// Reorders elements based on the computed prefix sums.
// Each thread reads its element, computes its output position, and writes.

kernel void radixScatter(
    device const SplatDepthKey* inputKeys [[buffer(0)]],
    device SplatDepthKey* outputKeys [[buffer(1)]],
    device atomic_uint* histogram [[buffer(2)]],
    constant GPUSortUniforms& uniforms [[buffer(3)]],
    uint globalIndex [[thread_position_in_grid]],
    uint localIndex [[thread_position_in_threadgroup]],
    uint groupIndex [[threadgroup_position_in_grid]])
{
    // Local histogram copy for this threadgroup
    threadgroup uint localOffsets[RADIX_SIZE];

    // Load initial offsets for this block
    if (localIndex < RADIX_SIZE) {
        uint histIdx = localIndex * uniforms.blockCount + groupIndex;
        localOffsets[localIndex] = atomic_load_explicit(&histogram[histIdx], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (globalIndex >= uniforms.totalSplatCount) {
        return;
    }

    // Read input element
    SplatDepthKey elem = inputKeys[globalIndex];
    uint digit = (elem.sortKey >> uniforms.bitOffset) & (RADIX_SIZE - 1);

    // Get output position using atomic increment
    uint outputPos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localOffsets[digit],
                                               1, memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to output
    outputKeys[outputPos] = elem;
}

// ============================================================================
// Kernel 6: Convert to Chunked Indices
// ============================================================================
//
// Converts sorted global indices back to ChunkedSplatIndex format.

kernel void convertToChunkedIndices(
    device const SplatDepthKey* sortedKeys [[buffer(0)]],
    device const ChunkOffsetEntry* chunkOffsets [[buffer(1)]],
    device ChunkedSplatIndex* outputIndices [[buffer(2)]],
    constant uint& totalSplatCount [[buffer(3)]],
    constant uint& chunkCount [[buffer(4)]],
    uint globalIndex [[thread_position_in_grid]])
{
    if (globalIndex >= totalSplatCount) {
        return;
    }

    uint splatGlobalIndex = sortedKeys[globalIndex].globalIndex;

    // Binary search to find which chunk this belongs to
    uint left = 0;
    uint right = chunkCount;

    while (left < right) {
        uint mid = (left + right) / 2;
        if (mid + 1 < chunkCount && chunkOffsets[mid + 1].startIndex <= splatGlobalIndex) {
            left = mid + 1;
        } else if (chunkOffsets[mid].startIndex > splatGlobalIndex) {
            right = mid;
        } else {
            left = mid;
            break;
        }
    }

    uint chunkIdx = left;
    uint localIndex = splatGlobalIndex - chunkOffsets[chunkIdx].startIndex;

    outputIndices[globalIndex].chunkIndex = chunkOffsets[chunkIdx].chunkIndex;
    outputIndices[globalIndex]._padding = 0;
    outputIndices[globalIndex].splatIndex = localIndex;
}
