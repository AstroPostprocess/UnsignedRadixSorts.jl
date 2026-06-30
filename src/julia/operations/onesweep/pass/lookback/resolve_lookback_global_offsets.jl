"""
    _resolve_lookback_global_offsets!(lookback, bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, ::Val{Pass})

Resolve each bucket's global scatter base through OneSweep decoupled lookback.

For each bucket, this helper walks backward from `tile_id - 1`, acquire-loads
lookback entries until they are nonzero, accumulates `_entry_count(entry)`, and
stops once a GLOBAL entry is reached. The final scatter base is
`bucket_offsets[Pass, bucket] + previous - local_offsets[bucket]`, written into
the worker's `global_offsets` slice. The helper then upgrades this tile's entry
to `_global_entry(previous + local_count)` with a release store.

CUB parallel: this combines `LoadBinsToOffsetsGlobal`,
`LookbackGlobal`, and `UpdateBinsGlobal`.

# Parameters

- `lookback`: Packed lookback table read for previous tiles and updated for this tile.
- `bucket_offsets`: Pass-wide 1-based bucket start offsets.
- `local_counts`: Per-worker bucket counts for the current tile.
- `local_offsets`: Per-worker exclusive digit prefixes for the current tile.
- `global_offsets`: Per-worker global scatter bases written in-place.
- `tile_id`: 0-based tile id claimed by the worker.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
@inline function _resolve_lookback_global_offsets!(lookback :: OffsetV, bucket_offsets :: OffsetV, local_counts :: OffsetV, local_offsets :: OffsetV, global_offsets :: OffsetV, tile_id :: Int, :: Val{Pass}) where {OffsetV <: Vector{UInt32}, Pass}
    worker_id = _worker_id()

    @inbounds for bucket in 1:256
        previous = zero(UInt32)
        prev_tile = tile_id - 1

        while prev_tile >= 0
            idx = _lookback_index(prev_tile, bucket)
            entry = Atomix.@atomic :acquire lookback[idx]

            while entry == zero(UInt32)
                entry = Atomix.@atomic :acquire lookback[idx]
            end

            previous += _entry_count(entry)

            if _is_global_entry(entry)
                break
            end

            prev_tile -= 1
        end

        idx_wb = _worker_bucket_index(worker_id, bucket)
        local_count = local_counts[idx_wb]
        idx_bo = _bucket_offsets_index(Pass, bucket)
        global_offsets[idx_wb] = bucket_offsets[idx_bo] + previous - local_offsets[idx_wb]

        idx_l = _lookback_index(tile_id, bucket)
        Atomix.@atomic :release lookback[idx_l] = _global_entry(previous + local_count)
    end

    return nothing
end
