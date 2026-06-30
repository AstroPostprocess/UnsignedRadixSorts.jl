"""
    _publish_lookback_partial!(lookback, local_counts, tile_id)

Publish this CUDA block's per-bucket counts as PARTIAL lookback entries.

Each thread publishes a strided subset of buckets. The helper reads the
block-local count, packs it with `_partial_entry`, and atomically stores it at
`_lookback_index(tile_id, bucket)`. Later blocks spin on these entries while
resolving their global offsets.

CUB parallel: this is the required `CountsCallback -> LookbackPartial`
publication of per-tile `bins`.

# Parameters

- `lookback`: CUDA packed lookback table updated with PARTIAL entries.
- `local_counts`: Shared-memory bucket counts to publish.
- `tile_id`: 0-based tile id claimed by the block.
"""
@inline function _publish_lookback_partial!(lookback :: OffsetV, local_counts :: OffsetV, tile_id :: Int) where {OffsetV <: CuDeviceVector{UInt32}}
    thread_id = Int(CUDA.threadIdx().x)
    nthreads = Int(CUDA.blockDim().x)

    bucket = thread_id
    while bucket <= 256
        @inbounds count = local_counts[bucket]
        idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
        entry = UnsignedRadixSorts._partial_entry(count)
        CUDA.atomic_xchg!(pointer(lookback, idx), entry)
        bucket += nthreads
    end
    CUDA.sync_threads()

    return nothing
end
