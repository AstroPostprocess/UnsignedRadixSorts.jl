using StaticArrays: MVector

"""
    _scatter_key_values_global!(keys_out, values_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup}, ::Val{Pass})

Scatter tile-ranked keys and paired permutation values.

Keys are first read from threadgroup memory in threadgroup-striped sorted order
and written to their global pass positions. Values are loaded in original input
order, reordered with the retained register ranks, and written using the sorted
key buckets.

CUB parallel: this combines `ScatterKeysGlobal()` with the value path in
`GatherScatterValues(ranks, bool_constant_v<KEYS_ONLY>)`: load values,
`ScatterValuesShared(values, ranks)`, then `ScatterValuesGlobal(digits)`.

# Parameters

- `keys_out`: Threadgroup-memory tile-ranked keys.
- `values_out`: Threadgroup-memory value staging storage for UInt8/UInt16 keys.
- `keys`: Per-thread cached keys in original SIMD-striped order.
- `ranks`: Per-thread stable tile-local ranks.
- `dst`: Active destination key buffer.
- `perm_src`: Active source permutation buffer.
- `perm_dst`: Active destination permutation buffer.
- `global_offsets`: Threadgroup-local per-bucket global scatter bases.
- `rangemin`: First 1-based source index in the tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerGroup}`: Compile-time Metal threadgroup size.
- `::Val{Pass}`: Compile-time radix pass selector.
"""
function _scatter_key_values_global! end

# UInt8/UInt16 need a separate UInt32 value-staging array because a key slot is
# narrower than a permutation value.
for KeyT in (UInt8, UInt16)
    @eval begin
        @inline function _scatter_key_values_global!(keys_out :: SharedK, values_out :: SharedV, keys :: MVector{ItemsPerThread, $KeyT}, ranks :: MVector{ItemsPerThread, UInt32}, dst :: KeyV, perm_src :: OffsetV, perm_dst :: OffsetV, global_offsets :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {ItemsPerThread, SharedK <: MtlDeviceVector{$KeyT}, SharedV <: MtlDeviceVector{UInt32}, KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            # Get this Metal thread's threadgroup and SIMD-group coordinates.
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

            # Cache each sorted item bucket for the later value write.
            sorted_buckets = MVector{ItemsPerThread, UInt16}(undef)

            # -----------------------------------------------------
            # 1. ScatterKeysGlobal: shared sorted keys -> global output.
            #
            # Keep the sorted key's bucket in a register so the later value
            # write can reuse the exact same global position calculation.
            item = 0
            while item < ItemsPerThread
                # Read the shared tile in sorted threadgroup-striped order.
                sorted_idx0 = item * ThreadsPerGroup + thread_id - 1
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
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # 2. LoadValues: read permutation values in original tile order.
            values = MVector{ItemsPerThread, UInt32}(undef)
            item = 0
            while item < ItemsPerThread
                # Load values using the original SIMD-striped input order.
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
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
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                if local_j <= tile_len
                    @inbounds values_out[Int(ranks[item + 1]) + 1] = values[item + 1]
                end
                item += 1
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # 4. ScatterValuesGlobal: sorted threads write values to the same
            # positions as the sorted keys.
            item = 0
            while item < ItemsPerThread
                # Revisit the same sorted slots used by ScatterKeysGlobal.
                sorted_idx0 = item * ThreadsPerGroup + thread_id - 1
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

# UInt32/UInt64 reuse the key staging array after every sorted key has been read.
# This preserves CUDA's shared-storage union without allocating another 8 KiB.
for KeyT in (UInt32, UInt64)
    @eval begin
        @inline function _scatter_key_values_global!(keys_out :: SharedK, keys :: MVector{ItemsPerThread, $KeyT}, ranks :: MVector{ItemsPerThread, UInt32}, dst :: KeyV, perm_src :: OffsetV, perm_dst :: OffsetV, global_offsets :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {ItemsPerThread, SharedK <: MtlDeviceVector{$KeyT}, SharedV <: MtlDeviceVector{UInt32}, KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            # Get this Metal thread's threadgroup and SIMD-group coordinates.
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

            # Cache each sorted item bucket for the later value write.
            sorted_buckets = MVector{ItemsPerThread, UInt16}(undef)

            # -----------------------------------------------------
            # 1. ScatterKeysGlobal: shared sorted keys -> global output.
            #
            # Keep the sorted key's bucket in a register so the later value
            # write can reuse the exact same global position calculation.
            item = 0
            while item < ItemsPerThread
                # Read the shared tile in sorted threadgroup-striped order.
                sorted_idx0 = item * ThreadsPerGroup + thread_id - 1
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
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # 2. LoadValues: read permutation values in original tile order.
            values = MVector{ItemsPerThread, UInt32}(undef)
            item = 0
            while item < ItemsPerThread
                # Load values using the original SIMD-striped input order.
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
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
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                if local_j <= tile_len
                    @inbounds keys_out[Int(ranks[item + 1]) + 1] = $KeyT(values[item + 1])
                end
                item += 1
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # 4. ScatterValuesGlobal: sorted threads write values to the same
            # positions as the sorted keys.
            item = 0
            while item < ItemsPerThread
                # Revisit the same sorted slots used by ScatterKeysGlobal.
                sorted_idx0 = item * ThreadsPerGroup + thread_id - 1
                if sorted_idx0 < tile_len
                    @inbounds bucket = Int(sorted_buckets[item + 1])
                    @inbounds output_idx = global_offsets[bucket] + UInt32(sorted_idx0)
                    @inbounds perm_dst[Int(output_idx)] = UInt32(keys_out[sorted_idx0 + 1])
                end
                item += 1
            end

            return nothing
        end
    end
end
