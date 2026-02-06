#include <metal_stdlib>
#include <simd/simd.h>
#include "ShaderCommon.h"

using namespace metal;

// ============================================================================
// GPU Bitonic Sort Implementation
// ============================================================================
//
// Bitonic sort uses compare-and-swap operations to sort data in parallel.
// It is deterministic (no atomics) and naturally suited to GPU execution.
//
// Pipeline:
//   1. computeDepths — compute depth keys from splat positions
//   2. bitonicSortLocal — sort 512-element blocks in shared memory (stages 0-8)
//   3. For stages 9..log2(N)-1:
//      a. bitonicMergeGlobal — compare-and-swap for large strides (≥512)
//      b. bitonicMergeLocal — finish stage in shared memory (strides <512)
//   4. convertToChunkedIndices — convert sorted indices back to chunk format
// ============================================================================

constant uint BLOCK_SIZE = 512;       // Elements per shared-memory block
constant uint HALF_BLOCK = 256;       // Threads per threadgroup
constant uint LOG2_BLOCK = 9;         // log2(512) = 9

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
// Kernel 2: Bitonic Sort Local (stages 0-8, shared memory)
// ============================================================================
//
// Each threadgroup sorts a block of 512 elements entirely in shared memory.
// 256 threads per group, each handles 2 elements.

kernel void bitonicSortLocal(
    device SplatDepthKey* depthKeys [[buffer(0)]],
    constant GPUSortUniforms& uniforms [[buffer(1)]],
    uint localId [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]])
{
    threadgroup SplatDepthKey shared_data[BLOCK_SIZE];

    uint blockStart = groupId * BLOCK_SIZE;
    uint idx0 = blockStart + localId;
    uint idx1 = blockStart + localId + HALF_BLOCK;

    // Load into shared memory, padding out-of-bounds with UINT_MAX
    if (idx0 < uniforms.paddedCount) {
        shared_data[localId] = depthKeys[idx0];
    } else {
        shared_data[localId] = SplatDepthKey{ UINT_MAX, UINT_MAX };
    }
    if (idx1 < uniforms.paddedCount) {
        shared_data[localId + HALF_BLOCK] = depthKeys[idx1];
    } else {
        shared_data[localId + HALF_BLOCK] = SplatDepthKey{ UINT_MAX, UINT_MAX };
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Run stages 0 through LOG2_BLOCK-1 (stages 0-8)
    for (uint stage = 0; stage < LOG2_BLOCK; stage++) {
        for (uint step = stage; step != UINT_MAX; step--) {
            uint stride = 1u << step;
            // Determine which pair this thread operates on
            uint pairIdx = localId + (localId / stride) * stride;
            uint partner = pairIdx + stride;

            if (partner < BLOCK_SIZE) {
                // Direction: ascending if the (stage+1)-th bit of the block-relative index is 0
                bool ascending = ((pairIdx >> (stage + 1)) & 1) == 0;

                SplatDepthKey a = shared_data[pairIdx];
                SplatDepthKey b = shared_data[partner];

                bool shouldSwap = ascending
                    ? (a.sortKey > b.sortKey)
                    : (a.sortKey < b.sortKey);

                if (shouldSwap) {
                    shared_data[pairIdx] = b;
                    shared_data[partner] = a;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back to global memory
    if (idx0 < uniforms.paddedCount) {
        depthKeys[idx0] = shared_data[localId];
    }
    if (idx1 < uniforms.paddedCount) {
        depthKeys[idx1] = shared_data[localId + HALF_BLOCK];
    }
}

// ============================================================================
// Kernel 3: Bitonic Merge Global (large strides ≥ 512)
// ============================================================================
//
// One compare-and-swap per thread for a single (stage, step) pair.

kernel void bitonicMergeGlobal(
    device SplatDepthKey* depthKeys [[buffer(0)]],
    constant GPUSortUniforms& uniforms [[buffer(1)]],
    uint globalId [[thread_position_in_grid]])
{
    uint stride = 1u << uniforms.bitonicStep;
    uint halfStride = stride;

    // Map thread to element pair
    uint pairIdx = globalId + (globalId / halfStride) * halfStride;
    uint partner = pairIdx + halfStride;

    if (partner >= uniforms.paddedCount) return;

    // Direction determined by the stage
    bool ascending = ((pairIdx >> (uniforms.bitonicStage + 1)) & 1) == 0;

    SplatDepthKey a = depthKeys[pairIdx];
    SplatDepthKey b = depthKeys[partner];

    bool shouldSwap = ascending
        ? (a.sortKey > b.sortKey)
        : (a.sortKey < b.sortKey);

    if (shouldSwap) {
        depthKeys[pairIdx] = b;
        depthKeys[partner] = a;
    }
}

// ============================================================================
// Kernel 4: Bitonic Merge Local (tail steps with stride < 512, shared memory)
// ============================================================================
//
// After bitonicMergeGlobal handles the large-stride step(s) of a stage,
// this kernel finishes the remaining steps (stride 256 down to 1) in shared memory.

kernel void bitonicMergeLocal(
    device SplatDepthKey* depthKeys [[buffer(0)]],
    constant GPUSortUniforms& uniforms [[buffer(1)]],
    uint localId [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]])
{
    threadgroup SplatDepthKey shared_data[BLOCK_SIZE];

    uint blockStart = groupId * BLOCK_SIZE;
    uint idx0 = blockStart + localId;
    uint idx1 = blockStart + localId + HALF_BLOCK;

    // Load into shared memory
    if (idx0 < uniforms.paddedCount) {
        shared_data[localId] = depthKeys[idx0];
    } else {
        shared_data[localId] = SplatDepthKey{ UINT_MAX, UINT_MAX };
    }
    if (idx1 < uniforms.paddedCount) {
        shared_data[localId + HALF_BLOCK] = depthKeys[idx1];
    } else {
        shared_data[localId + HALF_BLOCK] = SplatDepthKey{ UINT_MAX, UINT_MAX };
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Run steps from LOG2_BLOCK-1 (8) down to 0 within shared memory
    for (uint step = LOG2_BLOCK - 1; step != UINT_MAX; step--) {
        uint stride = 1u << step;
        uint pairIdx = localId + (localId / stride) * stride;
        uint partner = pairIdx + stride;

        if (partner < BLOCK_SIZE) {
            // Direction: use the global index for determining sort direction
            uint globalPairIdx = blockStart + pairIdx;
            bool ascending = ((globalPairIdx >> (uniforms.bitonicStage + 1)) & 1) == 0;

            SplatDepthKey a = shared_data[pairIdx];
            SplatDepthKey b = shared_data[partner];

            bool shouldSwap = ascending
                ? (a.sortKey > b.sortKey)
                : (a.sortKey < b.sortKey);

            if (shouldSwap) {
                shared_data[pairIdx] = b;
                shared_data[partner] = a;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (idx0 < uniforms.paddedCount) {
        depthKeys[idx0] = shared_data[localId];
    }
    if (idx1 < uniforms.paddedCount) {
        depthKeys[idx1] = shared_data[localId + HALF_BLOCK];
    }
}

// ============================================================================
// Kernel 5: Convert to Chunked Indices
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
