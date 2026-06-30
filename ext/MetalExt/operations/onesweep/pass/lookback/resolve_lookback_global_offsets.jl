"""
    _resolve_lookback_global_offsets!(lookback, bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, ::Val{Pass})

Resolve each bucket's Metal global scatter base through OneSweep decoupled
lookback.

Each lane owns a strided subset of buckets. For each bucket, it walks backward
from `tile_id - 1`, atomically loads lookback entries until they are nonzero,
accumulates `_entry_count(entry)`, and stops once a GLOBAL entry is reached. The
final scatter base is `bucket_offsets[Pass, bucket] + previous -
local_offsets[bucket]`, written into `global_offsets`. The helper then upgrades
this tile's entry to `_global_entry(previous + local_count)`.

CUB parallel: this combines `LoadBinsToOffsetsGlobal`,
`LookbackGlobal`, and `UpdateBinsGlobal`.

# Parameters

- `lookback`: Metal packed lookback table read for previous tiles and updated for this tile.
- `bucket_offsets`: Pass-wide 1-based bucket start offsets.
- `local_counts`: Threadgroup-memory bucket counts for the current tile.
- `local_offsets`: Threadgroup-memory exclusive digit prefixes for the current tile.
- `global_offsets`: Threadgroup-memory global scatter bases written in-place.
- `tile_id`: 0-based tile id claimed by the threadgroup.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
@inline function _resolve_lookback_global_offsets!(lookback :: OffsetV, bucket_offsets :: OffsetV, local_counts :: SharedV, local_offsets :: SharedV, global_offsets :: SharedV, tile_id :: Int, :: Val{Pass}) where {OffsetV <: MtlDeviceVector{UInt32}, SharedV <: MtlDeviceVector{UInt32}, Pass}
    lane_id = Int(Metal.thread_position_in_threadgroup().x)
    nlanes = Int(Metal.threads_per_threadgroup().x)

    bucket = lane_id
    while bucket <= 256
        previous = zero(UInt32)
        prev_tile = tile_id - 1

        while prev_tile >= 0
            idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket)
            entry = Metal.atomic_load_explicit(pointer(lookback, idx))

            while entry == zero(UInt32)
                entry = Metal.atomic_load_explicit(pointer(lookback, idx))
            end

            previous += UnsignedRadixSorts._entry_count(entry)

            if UnsignedRadixSorts._is_global_entry(entry)
                break
            end

            prev_tile -= 1
        end

        @inbounds begin
            local_count = local_counts[bucket]
            bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)]
            global_offsets[bucket] = bucket_start + previous - local_offsets[bucket]
        end

        idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket)
        global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
        Metal.atomic_store_explicit(pointer(lookback, idx_l), global_entry)

        bucket += nlanes
    end
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    return nothing
end
