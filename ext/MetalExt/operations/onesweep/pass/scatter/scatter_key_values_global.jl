"""
    _scatter_key_values_global!(src, dst, perm_src, perm_dst, global_offsets, local_ranks, rangemin, tile_len, ::Val{Pass})

Scatter Metal keys and their permutation values to final pass positions.

This helper computes the same scatter index as `_scatter_keys_global!`, writes
`src[i]` to `dst`, and writes the paired permutation value `perm_src[i]` to
`perm_dst` at the same index.

CUB parallel: this covers `ScatterKeysGlobal` plus the value path represented
by `GatherScatterValues`.

# Parameters

- `src`: Active Metal source key buffer for this pass.
- `dst`: Active Metal destination key buffer for this pass.
- `perm_src`: Active Metal source permutation buffer for this pass.
- `perm_dst`: Active Metal destination permutation buffer for this pass.
- `global_offsets`: Threadgroup-memory global scatter bases produced by lookback resolution.
- `local_ranks`: Threadgroup-memory tile-local ranks.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _scatter_key_values_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_key_values_global!(src :: KeyV, dst :: KeyV, perm_src :: OffsetV, perm_dst :: OffsetV, global_offsets, local_ranks, rangemin :: Int, tile_len :: Int, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, Pass}
            lane_id = Int(Metal.thread_position_in_threadgroup().x)
            nlanes = Int(Metal.threads_per_threadgroup().x)
            local_i = lane_id
            while local_i <= tile_len
                i = rangemin + local_i - 1
                @inbounds begin
                    bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    scatter_idx = global_offsets[bucket] + local_ranks[local_i]
                    dst[Int(scatter_idx)] = src[i]
                    perm_dst[Int(scatter_idx)] = perm_src[i]
                end
                local_i += nlanes
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            return nothing
        end
    end
end
