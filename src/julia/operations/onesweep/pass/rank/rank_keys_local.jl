"""
    _rank_keys_early_counts!(keys, rank_cursors, local_counts, local_offsets, global_offsets, lookback, local_ranks, tile_id, tile_len, ::Val{TileSize}, ::Val{Pass})

Compute stable tile-local ranks and publish early per-bucket counts.

This mirrors CUB `BlockRadixRankMatchEarlyCounts::RankKeys`: build per-tile
`bins` from cached `keys`, call the `CountsCallback` path to publish
LookbackPartial, scan bins into `exclusive_digit_prefix`, and write stable
0-based `ranks`. The CPU implementation keeps the same dataflow with serial
per-worker loops.

CUB reference: `BlockRadixRankMatchEarlyCounts::RankKeys` with
`CountsCallback`.

# Parameters

- `keys`: Per-worker cached keys produced by `_load_keys!`.
- `rank_cursors`: Per-worker cursor storage corresponding to CUB rank temp storage.
- `local_counts`: Per-worker bucket counts written in-place.
- `local_offsets`: Per-worker exclusive digit prefixes written in-place.
- `global_offsets`: Per-worker global offset storage later filled by lookback resolution.
- `lookback`: Packed global OneSweep lookback table.
- `local_ranks`: Per-worker tile-local ranks written in-place.
- `tile_id`: 0-based tile id claimed by the worker.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.

# Returns

The same `local_ranks` temporary buffer, now holding 0-based stable ranks.
"""
function _rank_keys_early_counts! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _rank_keys_early_counts!(keys :: KeyV, rank_cursors :: OffsetV, local_counts :: OffsetV, local_offsets :: OffsetV, global_offsets :: OffsetV, lookback :: OffsetV, local_ranks :: OffsetV, tile_id :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)

            # -----------------------------------------------------
            # 1. RankKeys(): clear and compute per-tile bins.
            #
            # CUB first builds per-warp digit counts and reduces them into
            # `bins`. A CPU worker owns the whole tile, so a serial histogram
            # over cached keys produces the same `local_counts` payload. Only
            # bins must be cleared here; exclusive_digit_prefix,
            # global_offsets, and rank cursors are overwritten by later stages.
            @inbounds for bucket in 1:256
                idx_wb = _worker_bucket_index(worker_id, bucket)
                local_counts[idx_wb] = zero(UInt32)
            end

            @inbounds for local_j in 1:tile_len
                key = keys[tile_base + local_j]
                bucket = _radix_bucket(key, Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                local_counts[idx_wb] += one(UInt32)
            end

            # -----------------------------------------------------
            # 2. CountsCallback -> LookbackPartial.
            #
            # Publish PARTIAL at the CountsCallback point, immediately after
            # RankKeys knows this tile's `bins`.
            _publish_lookback_partial!(lookback, local_counts, tile_id)

            # -----------------------------------------------------
            # 3. RankKeys(): bins -> exclusive_digit_prefix.
            #
            # Serially scan the 256 buckets into `local_offsets`.
            running = zero(UInt32)

            @inbounds for bucket in 1:256
                idx_wb = _worker_bucket_index(worker_id, bucket)
                local_offsets[idx_wb] = running
                running += local_counts[idx_wb]
            end

            # -----------------------------------------------------
            # 4. RankKeys(): initialize bucket cursors from prefixes.
            @inbounds for bucket in 1:256
                idx_wb = _worker_bucket_index(worker_id, bucket)
                rank_cursors[idx_wb] = local_offsets[idx_wb]
            end

            # -----------------------------------------------------
            # 5. RankKeys(): assign stable per-item ranks.
            #
            # Walking cached keys in tile order preserves the same stable rank
            # order CUB gets from ComputeRanksItem.
            @inbounds for local_j in 1:tile_len
                rank_idx = tile_base + local_j
                bucket = _radix_bucket(keys[rank_idx], Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                rank = rank_cursors[idx_wb]
                local_ranks[rank_idx] = rank
                rank_cursors[idx_wb] = rank + one(UInt32)
            end

            return local_ranks
        end
    end
end

"""
    _rank_keys_local!(src, local_counts, local_offsets, rank_cursors, local_ranks, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Build exclusive digit prefixes and stable tile-local ranks.

Legacy pre-`keys_out` helper. This helper first scans the worker's 256 local bucket counts into
`local_offsets`. It copies those offsets into `rank_cursors`, then walks the
tile in source order. Each item receives the current cursor for its bucket in
`local_ranks[rank_idx]`, and the cursor is incremented. This preserves
stability within the tile.

CUB parallel: the current split path uses `RankKeys` to produce
`exclusive_digit_prefix` and `ranks`, then `_scatter_keys_shared!` to stage
keys. This retained helper represents the older fused path that scattered
directly from the source tile.

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
