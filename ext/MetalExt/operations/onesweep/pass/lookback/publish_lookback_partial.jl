"""
    _publish_lookback_partial!(lookback, local_counts, tile_id)

Publish this Metal threadgroup's per-bucket counts as PARTIAL lookback entries.

Threads cooperatively publish the 256 buckets in strided order. Atomic exchange
is retained as the verified publication mechanism.

CUB parallel: `CountsCallback -> LookbackPartial`. The packed entry stores
`bins[bin] | LOOKBACK_PARTIAL_MASK` at
`d_lookback[block_idx * RADIX_DIGITS + bin]`, allowing later tiles to start
their decoupled lookback before this tile has finished global offset resolution.

# Parameters

- `lookback`: Packed global OneSweep lookback table.
- `local_counts`: Threadgroup-memory tile counts for all radix buckets.
- `tile_id`: Current 0-based tile id.
"""
@inline function _publish_lookback_partial!(lookback :: OffsetV, local_counts :: SharedV, tile_id :: Int) where {OffsetV <: MtlDeviceVector{UInt32}, SharedV <: MtlDeviceVector{UInt32}}
    # Threads cooperatively own buckets in threadgroup-strided order.
    thread_id = Int(Metal.thread_position_in_threadgroup().x)
    nthreads = Int(Metal.threads_per_threadgroup().x)

    bucket = thread_id
    while bucket <= 256
        # Pack this tile's local count with the PARTIAL state bit.
        @inbounds count = local_counts[bucket]
        idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
        entry = UnsignedRadixSorts._partial_entry(count)
        # Publish the count and state bit together. Later threadgroups spin
        # until this entry is nonzero, then accumulate the payload count.
        Metal.atomic_exchange_explicit(pointer(lookback, idx), entry)
        bucket += nthreads
    end

    # Make PARTIAL entries visible before this threadgroup depends on lookback state.
    Metal.threadgroup_barrier(Metal.MemoryFlagDevice | Metal.MemoryFlagThreadGroup)

    return nothing
end
