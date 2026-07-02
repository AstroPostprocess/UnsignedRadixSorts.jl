"""
    _clear_tile_storage!(local_counts, local_offsets, rank_cursors, global_offsets)

Clear the legacy per-worker 256-bucket temporary storage.

Every worker owns a flattened 256-bucket slice. This helper computes
`_worker_bucket_index(worker_id, bucket)` for each bucket and zeroes the count,
local prefix, rank cursor, and global scatter base for that worker's next tile.

Legacy reference: the current CPU Process() split clears only the storage that
must be reset at each stage. This helper is retained for older fused paths that
clear `bins`, `exclusive_digit_prefix`, rank cursors, and
`TempStorage_::global_offsets` together.

# Parameters

- `local_counts`: Per-worker bucket counts; overwritten with zeros.
- `local_offsets`: Per-worker exclusive digit prefixes; overwritten with zeros.
- `rank_cursors`: Per-worker cursors used while assigning stable local ranks; overwritten with zeros.
- `global_offsets`: Per-worker global scatter bases; overwritten with zeros.
"""
@inline function _clear_tile_storage!(local_counts :: OffsetV, local_offsets :: OffsetV, rank_cursors :: OffsetV, global_offsets :: OffsetV) where {OffsetV <: Vector{UInt32}}
    worker_id = _worker_id()

    @inbounds for bucket in 1:256
        idx = _worker_bucket_index(worker_id, bucket)
        local_counts[idx] = zero(UInt32)
        local_offsets[idx] = zero(UInt32)
        rank_cursors[idx] = zero(UInt32)
        global_offsets[idx] = zero(UInt32)
    end

    return nothing
end
