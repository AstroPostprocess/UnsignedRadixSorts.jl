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
