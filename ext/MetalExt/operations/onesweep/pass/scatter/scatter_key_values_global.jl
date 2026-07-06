using StaticArrays: MVector

"""
    _scatter_key_values_global!(src, dst, perm_src, perm_dst, global_offsets, ranks, rangemin, tile_len, ::Val{Pass})

Scatter tile keys and paired permutation values with Metal re-read staging.

CUDAExt combines `ScatterKeysGlobal()` with the value path in
`GatherScatterValues(ranks, bool_constant_v<KEYS_ONLY>)`: it reads tile-ranked
keys from `keys_out`, reuses shared payload storage for values, scatters values
by the retained ranks, and writes values to the same positions as their keys.

MetalExt keeps the public helper name but skips both `keys_out` and
`values_out`.  Keys and permutation values are re-read from their original
source positions and written directly to the final pass positions computed from
`global_offsets[bucket] + ranks[local_i]`.

Why re-read?  The CUB/CUDA shared exchange relies on the path
`MVector ranks -> keys_out[rank] = key` and then a second shared exchange for
values.  That path is intentionally avoided on Metal because it was the unstable
staging point.  The re-read version preserves stable sortperm semantics while
keeping the OneSweep lookback path unchanged.

# Parameters

- `src`: Active source key buffer for this radix pass.
- `dst`: Active destination key buffer.
- `perm_src`: Active source permutation buffer.
- `perm_dst`: Active destination permutation buffer.
- `global_offsets`: Threadgroup per-bucket global scatter bases.
- `ranks`: Threadgroup tile-local ranks indexed by original tile-local input position.
- `rangemin`: First 1-based source index in the tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{Pass}`: Compile-time radix pass selector.
"""
function _scatter_key_values_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_key_values_global!(src :: KeyV, dst :: KeyV, perm_src :: OffsetV, perm_dst :: OffsetV, global_offsets :: SharedV, ranks :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, SharedV <: MtlDeviceVector{UInt32}, Pass}
            # Walk keys and permutation values in original tile order.
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)

            local_i = thread_id
            while local_i <= tile_len
                # Re-read the key and paired permutation value from their source
                # buffers. The same `output_idx` is used for both, so the
                # permutation follows the stable key order produced by RankKeys.
                i = rangemin + local_i - 1
                @inbounds begin
                    key = src[i]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    output_idx = global_offsets[bucket] + ranks[local_i]
                    dst[Int(output_idx)] = key
                    perm_dst[Int(output_idx)] = perm_src[i]
                end
                local_i += nthreads
            end

            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            return nothing
        end
    end
end
