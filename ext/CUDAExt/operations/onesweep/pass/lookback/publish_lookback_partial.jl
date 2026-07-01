"""
    _publish_lookback_partial!(lookback, local_counts, tile_id)

Publish this CUDA block's per-bucket counts as PARTIAL lookback entries.

Threads cooperatively publish the 256 buckets in strided order. Atomic exchange
is retained as the verified publication mechanism.

CUB parallel: `CountsCallback -> LookbackPartial`. The packed entry stores
`bins[bin] | LOOKBACK_PARTIAL_MASK` at
`d_lookback[block_idx * RADIX_DIGITS + bin]`, allowing later tiles to start
their decoupled lookback before this tile has finished global offset resolution.

# Parameters

- `lookback`: Packed global OneSweep lookback table.
- `local_counts`: Shared-memory tile counts for all radix buckets.
- `tile_id`: Current 0-based tile id.
"""
@inline function _publish_lookback_partial!(lookback :: OffsetV, local_counts :: SharedV, tile_id :: Int) where {OffsetV <: CuDeviceVector{UInt32}, SharedV <: CuDeviceVector{UInt32}}
    # Threads cooperatively own buckets in block-strided order.
    thread_id = Int(CUDA.threadIdx().x)
    nthreads = Int(CUDA.blockDim().x)

    bucket = thread_id
    while bucket <= 256
        # Pack this tile's local count with the PARTIAL state bit.
        @inbounds count = local_counts[bucket]
        idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
        entry = UnsignedRadixSorts._partial_entry(count)
        # Publish the count and state bit together. Later blocks spin until
        # this entry is nonzero, then accumulate the payload count.
        CUDA.atomic_xchg!(pointer(lookback, idx), entry)
        bucket += nthreads
    end

    # Make PARTIAL entries visible before this block depends on lookback state.
    CUDA.sync_threads()
    return nothing
end
