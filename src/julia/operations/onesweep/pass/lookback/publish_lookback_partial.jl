"""
    _publish_lookback_partial!(lookback, local_counts, tile_id)

Publish this tile's per-bucket counts as PARTIAL lookback entries.

The helper reads each bucket count from the worker's flattened `local_counts`
slice, packs it with `_partial_entry`, and release-stores it at
`_lookback_index(tile_id, bucket)`. Later tiles spin on these entries while
resolving their global offsets.

CUB parallel: this is the required `CountsCallback -> LookbackPartial`
publication of per-tile `bins`.

# Parameters

- `lookback`: Packed lookback table updated with PARTIAL entries.
- `local_counts`: Per-worker bucket counts to publish.
- `tile_id`: 0-based tile id claimed by the worker.
"""
@inline function _publish_lookback_partial!(lookback :: OffsetV, local_counts :: OffsetV, tile_id :: Int) where {OffsetV <: Vector{UInt32}}
    worker_id = _worker_id()

    @inbounds for bucket in 1:256
        idx_wb = _worker_bucket_index(worker_id, bucket)
        idx = _lookback_index(tile_id, bucket)
        Atomix.@atomic :release lookback[idx] = _partial_entry(local_counts[idx_wb])
    end

    return nothing
end
