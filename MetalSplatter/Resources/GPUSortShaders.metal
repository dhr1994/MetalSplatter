#include <metal_stdlib>
#include <simd/simd.h>
#include "ShaderCommon.h"

using namespace metal;

// ============================================================================
// GPU Radix Sort Implementation (Simplified)
// ============================================================================
//
// This implements a 4-pass radix sort for 32-bit depth keys, processing 8 bits
// per pass. Uses a simplified approach with global atomics for correctness.
//
// Each pass:
//   1. Compute global histogram (256 buckets)
//   2. Prefix sum on histogram
//   3. Scatter using atomic counters
// ============================================================================

constant uint RADIX_BITS = 8;
constant uint RADIX_SIZE = 256;  // 2^8 = 256 buckets
constant uint THREADGROUP_SIZE = 256;

// ============================================================================
// Float-to-Sortable-Uint Conversion
// ============================================================================

inline uint floatToSortableUint(float f) {
    uint u = as_type<uint>(f);
    uint mask = -int(u >> 31) | 0x80000000;
    return u ^ mask;
}

// ============================================================================
// Kernel 1: Compute Depth Keys
// ============================================================================

kernel void computeDepths(
    device const ChunkTable* chunkTable [[buffer(0)]],
    device SplatDepthKey* depthKeys [[buffer(1)]],
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]])
{
    if (globalIndex >= uniforms.totalSplatCount) {
        return;
    }

    // Find which chunk this global index belongs to
    device ChunkInfo* chunks = chunkTable->chunks;
    uint chunkCount = chunkTable->enabledChunkCount;

    uint runningCount = 0;
    uint localIndex = globalIndex;

    for (uint c = 0; c < chunkCount; c++) {
        uint chunkSplatCount = chunks[c].splatCount;
        if (localIndex < chunkSplatCount) {
            // Found the chunk
            device Splat* splats = chunks[c].splats;
            float3 position = float3(splats[localIndex].position);

            // Compute depth
            float depth;
            if (uniforms.sortByDistance) {
                float3 diff = position - float3(uniforms.cameraPosition);
                depth = dot(diff, diff);
            } else {
                depth = dot(position, float3(uniforms.cameraForward));
            }

            // Negate for back-to-front sorting, convert to sortable uint
            depthKeys[globalIndex].sortKey = floatToSortableUint(-depth);
            depthKeys[globalIndex].globalIndex = globalIndex;
            return;
        }
        localIndex -= chunkSplatCount;
    }
}

// ============================================================================
// Kernel 2: Radix Histogram (Global)
// ============================================================================
//
// Counts occurrences of each 8-bit digit. Output: 256 global counts.

kernel void radixHistogram(
    device const SplatDepthKey* inputKeys [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]])
{
    if (globalIndex >= uniforms.totalSplatCount) {
        return;
    }

    uint key = inputKeys[globalIndex].sortKey;
    uint digit = (key >> uniforms.bitOffset) & (RADIX_SIZE - 1);
    atomic_fetch_add_explicit(&histogram[digit], 1, memory_order_relaxed);
}

// ============================================================================
// Kernel 3: Prefix Sum (256 elements only)
// ============================================================================
//
// Performs exclusive prefix sum on the 256-element histogram.
// Single threadgroup, simple parallel scan.

kernel void prefixSumLocal(
    device uint* histogram [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],  // unused in this version
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint localIndex [[thread_position_in_threadgroup]])
{
    threadgroup uint shared[RADIX_SIZE];

    // Load histogram into shared memory
    shared[localIndex] = histogram[localIndex];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele parallel scan (inclusive)
    for (uint stride = 1; stride < RADIX_SIZE; stride *= 2) {
        uint val = 0;
        if (localIndex >= stride) {
            val = shared[localIndex - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[localIndex] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Convert to exclusive scan (shift right, insert 0)
    uint myVal = shared[localIndex];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localIndex == 0) {
        histogram[0] = 0;
    } else {
        histogram[localIndex] = shared[localIndex - 1];
    }
}

// ============================================================================
// Kernel 4: Add Block Sums (not needed in simplified version)
// ============================================================================

kernel void addBlockSums(
    device uint* histogram [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant GPUSortUniforms& uniforms [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]],
    uint localIndex [[thread_position_in_threadgroup]],
    uint groupIndex [[threadgroup_position_in_grid]])
{
    // Not used in simplified version - kept for API compatibility
}

// ============================================================================
// Kernel 5: Radix Scatter (with global atomics)
// ============================================================================
//
// Reorders elements based on digit values using atomic counters.

kernel void radixScatter(
    device const SplatDepthKey* inputKeys [[buffer(0)]],
    device SplatDepthKey* outputKeys [[buffer(1)]],
    device atomic_uint* histogram [[buffer(2)]],
    constant GPUSortUniforms& uniforms [[buffer(3)]],
    uint globalIndex [[thread_position_in_grid]])
{
    if (globalIndex >= uniforms.totalSplatCount) {
        return;
    }

    SplatDepthKey elem = inputKeys[globalIndex];
    uint digit = (elem.sortKey >> uniforms.bitOffset) & (RADIX_SIZE - 1);

    // Get unique output position using atomic increment
    uint outputPos = atomic_fetch_add_explicit(&histogram[digit], 1, memory_order_relaxed);

    outputKeys[outputPos] = elem;
}

// ============================================================================
// Kernel 6: Convert to Chunked Indices
// ============================================================================

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
