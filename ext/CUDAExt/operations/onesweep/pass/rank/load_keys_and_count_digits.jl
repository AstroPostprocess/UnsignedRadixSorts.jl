using StaticArrays: MVector

"""
    _load_keys!(src, rangemin, tile_len, ::Val{TileSize}, ::Val{ThreadsPerBlock})

Load one CUDA tile into fixed-size per-thread key storage.

Each thread owns `cld(TileSize, ThreadsPerBlock)` entries in warp-striped input
order. Valid keys are read from global memory exactly once. Tail entries are
filled with zero and ignored by later validity guards.

CUB parallel: this is the `LoadKeys(block_idx * TILE_ITEMS, keys)` part of
`AgentRadixSortOnesweep::Process()`. The count-producing work that older
versions did here is now fused into `_rank_keys_early_counts!`, matching
`BlockRadixRankT::RankKeys(... CountsCallback(...))`.

# Parameters

- `src`: Active CUDA source key buffer for this radix pass.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerBlock}`: Compile-time CUDA block size.

# Returns

- Fixed-size mutable vector containing this thread's cached keys.
"""
function _load_keys! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _load_keys!(src :: KeyV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerBlock}) where {KeyV <: CuDeviceVector{$KeyT}, TileSize, ThreadsPerBlock}
            ItemsPerThread = cld(TileSize, ThreadsPerBlock)
            keys = MVector{ItemsPerThread, $KeyT}(undef)

            # Get this CUDA thread's block and warp coordinates.
            thread_id = Int(CUDA.threadIdx().x)
            warp_threads = Int(CUDA.warpsize())
            warp_id = fld(thread_id - 1, warp_threads)
            lane_in_warp = (thread_id - 1) % warp_threads

            # Load this thread's register items in the same order RankKeys uses.
            item = 0
            while item < ItemsPerThread
                # Warp-striped ownership matches CUB's per-thread key array
                # order: warp, then item slot, then lane.
                local_j = warp_id * warp_threads * ItemsPerThread + item * warp_threads + lane_in_warp + 1
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

            return keys
        end
    end
end
