using StaticArrays: MVector

"""
    _load_keys!(keys, src, rangemin, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup})

Load one Metal tile into caller-owned fixed-size per-thread key storage.

Each thread owns `cld(TileSize, ThreadsPerGroup)` entries in SIMD-striped input
order. Valid keys are read from device memory exactly once. Tail entries are
filled with zero and ignored by later validity guards.

CUB parallel: this is the `LoadKeys(block_idx * TILE_ITEMS, keys)` part of
`AgentRadixSortOnesweep::Process()`. The count-producing work that older
versions did here is now fused into `_rank_keys_early_counts!`, matching
`BlockRadixRankT::RankKeys(... CountsCallback(...))`.

# Parameters

- `keys`: Caller-owned fixed-size mutable vector receiving cached keys.
- `src`: Active Metal source key buffer for this radix pass.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerGroup}`: Compile-time Metal threadgroup size.
"""
function _load_keys! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _load_keys!(keys :: MVector{ItemsPerThread, $KeyT}, src :: KeyV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}) where {ItemsPerThread, KeyV <: MtlDeviceVector{$KeyT}, TileSize, ThreadsPerGroup}
            # Get this Metal thread's threadgroup and SIMD-group coordinates.
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

            # Load this thread's register items in the same order RankKeys uses.
            item = 0
            while item < ItemsPerThread
                # SIMD-striped ownership matches CUB's per-thread key array
                # order: SIMD group, then item slot, then lane.
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                key = zero($KeyT)

                # Guard partial tiles; invalid tail lanes keep a dummy zero key.
                if local_j <= tile_len
                    i = rangemin + local_j - 1
                    @inbounds key = src[i]
                end

                # Store in per-thread register storage for rank/scatter reuse.
                @inbounds keys[item + 1] = key
                item += 1
            end

            return nothing
        end
    end
end
