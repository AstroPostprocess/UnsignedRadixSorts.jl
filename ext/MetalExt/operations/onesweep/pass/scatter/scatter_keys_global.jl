using StaticArrays: MVector

"""
    _scatter_keys_shared!(keys_out, keys, ranks, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup})

Reorder cached keys into stable tile-rank order in threadgroup memory.

CUB parallel: `ScatterKeysShared(keys, ranks)`. After `RankKeys` produces
tile-local ranks, CUB writes each key to `s.keys_out[rank]` so later global
scatter can read a digit-sorted tile.

# Parameters

- `keys_out`: Threadgroup-memory key staging array.
- `keys`: Per-thread cached keys in SIMD-striped input order.
- `ranks`: Per-thread stable tile-local ranks.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerGroup}`: Compile-time Metal threadgroup size.
"""
function _scatter_keys_shared! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_keys_shared!(keys_out :: SharedK, keys :: MVector{ItemsPerThread, $KeyT}, ranks :: MVector{ItemsPerThread, UInt32}, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}) where {ItemsPerThread, SharedK <: MtlDeviceVector{$KeyT}, TileSize, ThreadsPerGroup}
            # Get this Metal thread's SIMD-striped input coordinates.
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

            # Each valid input item writes once to its tile-local sorted rank.
            item = 0
            while item < ItemsPerThread
                # Match the local_j mapping used by `_load_keys!` and RankKeys.
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                if local_j <= tile_len
                    # `ranks` is 0-based like CUB's rank; Julia threadgroup-memory
                    # indexing is 1-based, so add one only at the array access.
                    @inbounds keys_out[Int(ranks[item + 1]) + 1] = keys[item + 1]
                end
                item += 1
            end

            # Ensure all sorted keys are visible before any global scatter reads them.
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            return nothing
        end
    end
end

"""
    _scatter_keys_global!(keys_out, dst, global_offsets, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup}, ::Val{Pass})

Write tile-ranked keys from threadgroup memory to final pass positions.

Threads read `keys_out` in threadgroup-striped sorted order. The sorted tile
index is combined with the per-bucket threadgroup-local global base, matching
CUB's `ScatterKeysGlobal` dataflow.

CUB parallel: `ScatterKeysGlobal()` computes
`idx + s.global_offsets[Digit(key)]` from the sorted shared tile and writes the
key to d_keys_out.

# Parameters

- `keys_out`: Threadgroup-memory tile-ranked keys.
- `dst`: Active destination key buffer.
- `global_offsets`: Threadgroup-local per-bucket global scatter bases.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerGroup}`: Compile-time Metal threadgroup size.
- `::Val{Pass}`: Compile-time radix pass selector.
"""
function _scatter_keys_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_keys_global!(keys_out :: SharedK, dst :: KeyV, global_offsets :: SharedV, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {SharedK <: MtlDeviceVector{$KeyT}, KeyV <: MtlDeviceVector{$KeyT}, SharedV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            # Use the same per-thread item count as the shared scatter stage.
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            ItemsPerThread = cld(TileSize, ThreadsPerGroup)

            item = 0
            while item < ItemsPerThread
                # Global scatter walks the shared tile in threadgroup-striped
                # sorted order, matching CUB's idx = threadIdx.x + u * BLOCK_THREADS.
                sorted_idx0 = item * ThreadsPerGroup + thread_id - 1
                if sorted_idx0 < tile_len
                    # Recompute the digit from the sorted key to select its
                    # lookback-resolved global bucket base.
                    @inbounds key = keys_out[sorted_idx0 + 1]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    @inbounds output_idx = global_offsets[bucket] + UInt32(sorted_idx0)
                    @inbounds dst[Int(output_idx)] = key
                end
                item += 1
            end

            return nothing
        end
    end
end
