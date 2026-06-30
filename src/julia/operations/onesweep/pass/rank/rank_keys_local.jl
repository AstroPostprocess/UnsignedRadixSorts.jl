"""
    _rank_keys_local!(src, local_counts, local_offsets, rank_cursors, local_ranks, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Build exclusive digit prefixes and stable tile-local ranks.

This helper first scans the worker's 256 local bucket counts into
`local_offsets`. It copies those offsets into `rank_cursors`, then walks the
tile in source order. Each item receives the current cursor for its bucket in
`local_ranks[rank_idx]`, and the cursor is incremented. This preserves
stability within the tile.

CUB parallel: `RankKeys` returns `exclusive_digit_prefix` and `ranks`;
`ScatterKeysShared` is represented here by retaining `local_ranks` instead of
physically staging keys in shared memory.

# Parameters

- `src`: Active source key buffer for this pass.
- `local_counts`: Per-worker bucket counts produced by `_load_keys_and_count_digits!`.
- `local_offsets`: Per-worker exclusive digit prefixes written in-place.
- `rank_cursors`: Per-worker temporary cursors overwritten while assigning ranks.
- `local_ranks`: Per-worker tile-local ranks written in-place.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _rank_keys_local! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _rank_keys_local!(src :: KeyV, local_counts :: OffsetV, local_offsets :: OffsetV, rank_cursors :: OffsetV, local_ranks :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)
            rangemax = rangemin + tile_len - 1

            # ####################################################
            # 1. RankKeys(): bins -> exclusive_digit_prefix.
            #
            # CUB BlockRadixRank scans the per-tile `bins` to produce
            # `exclusive_digit_prefix`. In the CPU implementation,
            # `local_counts` already contains this worker's tile histogram, so
            # scan its 256-bucket slice into `local_offsets`.
            running = zero(UInt32)

            @inbounds for bucket in 1:256
                idx_wb = _worker_bucket_index(worker_id, bucket)
                local_offsets[idx_wb] = running
                running += local_counts[idx_wb]
            end

            # ####################################################
            # 2. RankKeys(): initialize rank cursors from exclusive prefixes.
            #
            # CUB's shared-memory scatter path consumes the ranks produced by
            # RankKeys. Here we represent the same staging by keeping a cursor
            # for each bucket, seeded from `exclusive_digit_prefix`, and writing
            # tile-local ranks into `local_ranks`.
            @inbounds for bucket in 1:256
                idx_wb = _worker_bucket_index(worker_id, bucket)
                rank_cursors[idx_wb] = local_offsets[idx_wb]
            end

            # ####################################################
            # 3. RankKeys(): assign stable per-item local ranks.
            #
            # Walk the tile in source order. Each item receives the current
            # cursor for its digit, then the cursor advances. Because the tile
            # is visited in input order, equal digits retain stable ordering.
            @inbounds for i in rangemin:rangemax
                rank_idx = tile_base + i - rangemin + 1
                bucket = _radix_bucket(src[i], Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                rank = rank_cursors[idx_wb]
                local_ranks[rank_idx] = rank
                rank_cursors[idx_wb] = rank + one(UInt32)
            end

            return nothing
        end
    end
end
