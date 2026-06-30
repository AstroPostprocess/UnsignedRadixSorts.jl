"""
    _resolve_lookback_global_offsets!(lookback, bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, ::Val{Pass})

Resolve each bucket's CUDA global scatter base through OneSweep decoupled
lookback.

Each thread owns a strided subset of buckets. For each bucket, it walks backward
from `tile_id - 1`, atomically loads lookback entries until they are nonzero,
accumulates `_entry_count(entry)`, and stops once a GLOBAL entry is reached. The
final scatter base is `bucket_offsets[Pass, bucket] + previous -
local_offsets[bucket]`, written into `global_offsets`. The helper then upgrades
this tile's entry to `_global_entry(previous + local_count)`.

CUB parallel: this combines `LoadBinsToOffsetsGlobal`,
`LookbackGlobal`, and `UpdateBinsGlobal`.

# Parameters

- `lookback`: CUDA packed lookback table read for previous tiles and updated for this tile.
- `bucket_offsets`: Pass-wide 1-based bucket start offsets.
- `local_counts`: Shared-memory bucket counts for the current tile.
- `local_offsets`: Shared-memory exclusive digit prefixes for the current tile.
- `global_offsets`: Shared-memory global scatter bases written in-place.
- `tile_id`: 0-based tile id claimed by the block.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
@inline function _resolve_lookback_global_offsets!(lookback :: OffsetV, bucket_offsets :: OffsetV, local_counts :: SharedV, local_offsets :: SharedV, global_offsets :: SharedV, tile_id :: Int, :: Val{Pass}) where {OffsetV <: CuDeviceVector{UInt32}, SharedV <: CuDeviceVector{UInt32}, Pass}
    thread_id = Int(CUDA.threadIdx().x)
    nthreads = Int(CUDA.blockDim().x)

    bucket = thread_id
    while bucket <= 256
        previous = zero(UInt32)
        prev_tile = tile_id - 1

        while prev_tile >= 0
            idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket)
            entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))

            while entry == zero(UInt32)
                entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))
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
        CUDA.atomic_xchg!(pointer(lookback, idx_l), global_entry)

        bucket += nthreads
    end
    CUDA.sync_threads()

    return nothing
end
