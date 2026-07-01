"""
    _resolve_lookback_global_offsets!(lookback, bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, ::Val{Pass})

Resolve all radix buckets' global scatter bases through decoupled lookback.

Threads cooperatively own the 256 buckets in strided order, so the helper works
for any supported whole-warp block size from 32 through 256 threads.

CUB parallel: this combines `LoadBinsToOffsetsGlobal`,
`LookbackGlobal(bins)`, and `UpdateBinsGlobal(bins, exclusive_digit_prefix)`.
For each bucket, the block seeds a scatter base from pass-wide bucket offsets,
walks backward through d_lookback until it reaches a GLOBAL prefix, then
upgrades this tile's PARTIAL entry to GLOBAL.

# Parameters

- `lookback`: Packed global lookback table.
- `bucket_offsets`: Pass-wide 1-based radix bucket starts.
- `local_counts`: Shared tile counts for all radix buckets.
- `local_offsets`: Shared exclusive tile prefixes for all radix buckets.
- `global_offsets`: Shared global scatter bases written by this helper.
- `tile_id`: Current 0-based tile id.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
@inline function _resolve_lookback_global_offsets!(lookback :: OffsetV, bucket_offsets :: OffsetV, local_counts :: SharedV, local_offsets :: SharedV, global_offsets :: SharedV, tile_id :: Int, :: Val{Pass}) where {OffsetV <: CuDeviceVector{UInt32}, SharedV <: CuDeviceVector{UInt32}, Pass}
    # Threads cooperatively own buckets in block-strided order.
    thread_id = Int(CUDA.threadIdx().x)
    nthreads = Int(CUDA.blockDim().x)

    bucket = thread_id
    while bucket <= 256
        # LookbackGlobal: accumulate same-bucket counts from preceding tiles.
        # PARTIAL entries contribute only their local count; the first GLOBAL
        # entry contributes a complete prefix and terminates the walk.
        previous = zero(UInt32)
        prev_tile = tile_id - 1

        while prev_tile >= 0
            # Read the previous tile's entry for this same radix bucket.
            idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket)

            # Retain the verified atomic RMW polling path until a true volatile
            # non-RMW load has been validated in generated PTX.
            entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))
            while entry == zero(UInt32)
                entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))
            end

            # Add the entry payload; GLOBAL means the prefix is complete.
            previous += UnsignedRadixSorts._entry_count(entry)
            UnsignedRadixSorts._is_global_entry(entry) && break
            prev_tile -= 1
        end

        @inbounds begin
            local_count = local_counts[bucket]
            bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)]
            # LoadBinsToOffsetsGlobal seeds the pass-wide bucket base and
            # subtracts this tile's exclusive digit prefix. Adding `previous`
            # shifts the tile to its final global position.
            global_offsets[bucket] = bucket_start + previous - local_offsets[bucket]
        end

        idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket)
        global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
        # UpdateBinsGlobal: publish a complete prefix for later tiles.
        CUDA.atomic_xchg!(pointer(lookback, idx_l), global_entry)

        # Move to this thread's next bucket.
        bucket += nthreads
    end

    # All buckets' global_offsets are ready for the following scatter stage.
    CUDA.sync_threads()
    return nothing
end
