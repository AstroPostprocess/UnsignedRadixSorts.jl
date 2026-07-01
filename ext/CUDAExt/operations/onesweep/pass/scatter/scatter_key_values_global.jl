using StaticArrays: MVector

"""
    _scatter_key_values_global!(keys_out, values_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, ::Val{TileSize}, ::Val{ThreadsPerBlock}, ::Val{Pass})

Scatter tile-ranked keys and paired permutation values.

Keys are first read from shared memory in block-striped sorted order and written
to their global pass positions. After all key reads finish, the same dynamic
shared-memory region is reinterpreted as `UInt32` value staging storage. Values
are loaded in original input order, reordered with the retained register ranks,
and written using the sorted key buckets.

CUB parallel: this combines `ScatterKeysGlobal()` with the value path in
`GatherScatterValues(ranks, bool_constant_v<KEYS_ONLY>)`: load values,
`ScatterValuesShared(values, ranks)`, then `ScatterValuesGlobal(digits)`.

# Parameters

- `keys_out`: Dynamic shared-memory tile-ranked keys.
- `values_out`: Dynamic shared-memory value staging storage, sharing the keys
  payload region after key scatter is finished.
- `keys`: Per-thread cached keys in original warp-striped order.
- `ranks`: Per-thread stable tile-local ranks.
- `dst`: Active destination key buffer.
- `perm_src`: Active source permutation buffer.
- `perm_dst`: Active destination permutation buffer.
- `global_offsets`: Shared per-bucket global scatter bases.
- `rangemin`: First 1-based source index in the tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerBlock}`: Compile-time CUDA block size.
- `::Val{Pass}`: Compile-time radix pass selector.
"""
function _scatter_key_values_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_key_values_global!(keys_out :: SharedK, values_out :: SharedV, keys :: MVector{ItemsPerThread, $KeyT}, ranks :: MVector{ItemsPerThread, UInt32}, dst :: KeyV, perm_src :: OffsetV, perm_dst :: OffsetV, global_offsets :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerBlock}, :: Val{Pass}) where {ItemsPerThread, SharedK <: CuDeviceVector{$KeyT}, KeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, SharedV <: CuDeviceVector{UInt32}, TileSize, ThreadsPerBlock, Pass}
            # Get this CUDA thread's block and warp coordinates.
            thread_id = Int(CUDA.threadIdx().x)
            warp_threads = Int(CUDA.warpsize())
            warp_id = fld(thread_id - 1, warp_threads)
            lane_in_warp = (thread_id - 1) % warp_threads

            # Cache each sorted item bucket for the later value write.
            sorted_buckets = MVector{ItemsPerThread, UInt16}(undef)

            # -----------------------------------------------------
            # 1. ScatterKeysGlobal: shared sorted keys -> global output.
            #
            # Keep the sorted key's bucket in a register so the later value
            # write can reuse the exact same global position calculation.
            item = 0
            while item < ItemsPerThread
                # Read the shared tile in sorted block-striped order.
                sorted_idx0 = item * ThreadsPerBlock + thread_id - 1
                bucket = 1

                if sorted_idx0 < tile_len
                    @inbounds key = keys_out[sorted_idx0 + 1]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    @inbounds output_idx = global_offsets[bucket] + UInt32(sorted_idx0)
                    @inbounds dst[Int(output_idx)] = key
                end

                @inbounds sorted_buckets[item + 1] = UInt16(bucket)
                item += 1
            end

            # All key reads must finish before the dynamic shared region is
            # reused as UInt32 value staging storage. This mirrors CUB's union
            # between keys_out and values_out in TempStorage_.
            CUDA.sync_threads()

            # 2. LoadValues: read permutation values in original tile order.
            values = MVector{ItemsPerThread, UInt32}(undef)
            item = 0
            while item < ItemsPerThread
                # Load values using the original warp-striped input order.
                local_j = warp_id * warp_threads * ItemsPerThread + item * warp_threads + lane_in_warp + 1
                value = zero(UInt32)

                if local_j <= tile_len
                    i = rangemin + local_j - 1
                    @inbounds value = perm_src[i]
                end

                @inbounds values[item + 1] = value
                item += 1
            end

            # 3. ScatterValuesShared: original input threads reorder values with
            # the same ranks that were used by ScatterKeysShared.
            item = 0
            while item < ItemsPerThread
                # Reuse the key rank so values follow the stable key order.
                local_j = warp_id * warp_threads * ItemsPerThread + item * warp_threads + lane_in_warp + 1
                if local_j <= tile_len
                    @inbounds values_out[Int(ranks[item + 1]) + 1] = values[item + 1]
                end
                item += 1
            end
            CUDA.sync_threads()

            # 4. ScatterValuesGlobal: sorted threads write values to the same
            # positions as the sorted keys.
            item = 0
            while item < ItemsPerThread
                # Revisit the same sorted slots used by ScatterKeysGlobal.
                sorted_idx0 = item * ThreadsPerBlock + thread_id - 1
                if sorted_idx0 < tile_len
                    @inbounds bucket = Int(sorted_buckets[item + 1])
                    @inbounds output_idx = global_offsets[bucket] + UInt32(sorted_idx0)
                    @inbounds perm_dst[Int(output_idx)] = values_out[sorted_idx0 + 1]
                end
                item += 1
            end

            return nothing
        end
    end
end
