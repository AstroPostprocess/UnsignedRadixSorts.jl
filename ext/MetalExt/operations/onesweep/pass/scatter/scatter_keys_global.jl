using StaticArrays: MVector

"""
    _scatter_keys_shared!(keys_out, keys, ranks, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup})

Reorder cached keys into stable tile-rank order in threadgroup memory.

CUB parallel: `ScatterKeysShared(keys, ranks)`.
"""
function _scatter_keys_shared! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_keys_shared!(keys_out :: SharedK, keys :: MVector{ItemsPerThread, $KeyT}, ranks :: MVector{ItemsPerThread, UInt32}, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}) where {ItemsPerThread, SharedK <: MtlDeviceVector{$KeyT}, TileSize, ThreadsPerGroup}
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

            item = 0
            while item < ItemsPerThread
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                if local_j <= tile_len
                    @inbounds keys_out[Int(ranks[item + 1]) + 1] = keys[item + 1]
                end
                item += 1
            end

            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            return nothing
        end
    end
end

"""
    _scatter_keys_global!(keys_out, dst, global_offsets, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup}, ::Val{Pass})

Write tile-ranked keys from threadgroup memory to final pass positions.

CUB parallel: `ScatterKeysGlobal()`.
"""
function _scatter_keys_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_keys_global!(keys_out :: SharedK, dst :: KeyV, global_offsets :: SharedV, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {SharedK <: MtlDeviceVector{$KeyT}, KeyV <: MtlDeviceVector{$KeyT}, SharedV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            ItemsPerThread = cld(TileSize, ThreadsPerGroup)

            item = 0
            while item < ItemsPerThread
                sorted_idx0 = item * ThreadsPerGroup + thread_id - 1
                if sorted_idx0 < tile_len
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
