# ========================================================================== #
#  Test: OneSweep pass RankKeys and CountsCallback
# ========================================================================== #
#
#  What this file tests
#  1. RankKeys
#     - `_rank_keys_early_counts!` builds per-tile `bins`, scans them into
#       `exclusive_digit_prefix`, and assigns stable CUB-style 0-based ranks.
#
#  2. CountsCallback
#     - The same helper publishes PARTIAL lookback entries immediately after
#       the tile counts are known.
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

@testset "OneSweep pass RankKeys with CountsCallback" begin
    T = UInt16
    tile_size = 8
    pass = 1
    src = T[0x02, 0x01, 0x02, 0x00, 0x01, 0x02]

    ranked = onesweep_on_default_thread() do
        ws = onesweep_pass_workspace(T, length(src), tile_size)
        tile_base = onesweep_pass_worker_base(tile_size)

        # Process() first runs LoadKeys; RankKeys consumes the cached tile.
        ONESWEEP_PASS_URS._load_keys!(src, ws.keys, 1, length(src), Val(tile_size))

        # RankKeys computes bins, prefixes, local ranks, and PARTIAL entries.
        ranks = ONESWEEP_PASS_URS._rank_keys_early_counts!(
            ws.keys,
            ws.rank_cursors,
            ws.local_counts,
            ws.local_offsets,
            ws.global_offsets,
            ws.lookback,
            ws.local_ranks,
            0,
            length(src),
            Val(tile_size),
            Val(pass),
        )

        return (
            same_buffer = ranks === ws.local_ranks,
            # Ranks are in original tile order, using CUB's 0-based convention.
            ranks = copy(ws.local_ranks[(tile_base + 1):(tile_base + length(src))]),
            counts = UInt32[
                ws.local_counts[onesweep_pass_worker_bucket(1)],
                ws.local_counts[onesweep_pass_worker_bucket(2)],
                ws.local_counts[onesweep_pass_worker_bucket(3)],
            ],
            offsets = UInt32[
                ws.local_offsets[onesweep_pass_worker_bucket(1)],
                ws.local_offsets[onesweep_pass_worker_bucket(2)],
                ws.local_offsets[onesweep_pass_worker_bucket(3)],
            ],
            entries = UInt32[
                ws.lookback[ONESWEEP_PASS_URS._lookback_index(0, 1)],
                ws.lookback[ONESWEEP_PASS_URS._lookback_index(0, 2)],
                ws.lookback[ONESWEEP_PASS_URS._lookback_index(0, 3)],
                ws.lookback[ONESWEEP_PASS_URS._lookback_index(0, 4)],
            ],
        )
    end

    @test ranked.same_buffer
    @test ranked.ranks == UInt32[3, 1, 4, 0, 2, 5]
    @test ranked.counts == UInt32[1, 2, 3]
    @test ranked.offsets == UInt32[0, 1, 3]

    # Buckets 0, 1, and 2 publish their counts; bucket 3 publishes an empty bin.
    for (entry, count) in zip(ranked.entries, UInt32[1, 2, 3, 0])
        @test ONESWEEP_PASS_URS._is_partial_entry(entry)
        @test ONESWEEP_PASS_URS._entry_count(entry) == count
    end
end
