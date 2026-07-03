"""
    _publish_lookback_partial!(lookback, local_counts, tile_id)

Publish this Metal threadgroup's per-bucket counts as PARTIAL lookback entries.

Each lane publishes a strided subset of buckets. The helper reads the
threadgroup-local count, packs it with `_partial_entry`, and atomically
exchanges it at `_lookback_index(tile_id, bucket)`. Later threadgroups spin on
these entries while resolving their global offsets.

CUB parallel: this is the required `CountsCallback -> LookbackPartial`
publication of per-tile `bins`.

# Parameters

- `lookback`: Metal packed lookback table updated with PARTIAL entries.
- `local_counts`: Threadgroup-memory bucket counts to publish.
- `tile_id`: 0-based tile id claimed by the threadgroup.
"""
@inline function _publish_lookback_partial!(lookback :: OffsetV, local_counts :: SharedV, tile_id :: Int) where {OffsetV <: MtlDeviceVector{UInt32}, SharedV <: MtlDeviceVector{UInt32}}
    lane_id = Int(Metal.thread_position_in_threadgroup().x)
    nlanes = Int(Metal.threads_per_threadgroup().x)

    bucket = lane_id
    while bucket <= 256
        @inbounds count = local_counts[bucket]
        idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
        entry = UnsignedRadixSorts._partial_entry(count)
        Metal.atomic_exchange_explicit(pointer(lookback, idx), entry)
        bucket += nlanes
    end
    Metal.threadgroup_barrier(Metal.MemoryFlagDevice | Metal.MemoryFlagThreadGroup)

    return nothing
end
