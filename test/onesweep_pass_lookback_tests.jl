# ========================================================================== #
#  Test: OneSweep pass LookbackGlobal
# ========================================================================== #
#
#  What this file tests
#  1. LookbackGlobal
#     - `_resolve_lookback_global_offsets!` walks prior tile entries, combines
#       them with pass-wide bucket starts, and writes per-bucket scatter bases.
#
#  2. UpdateBinsGlobal
#     - The current tile's lookback entries are upgraded from PARTIAL payloads
#       to GLOBAL prefix payloads for later tiles.
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

@testset "OneSweep pass LookbackGlobal" begin
    T = UInt32
    tile_size = 4
    pass = 1

    looked_back = onesweep_on_default_thread() do
        ws = onesweep_pass_workspace(T, 8, tile_size)

        # Tile 0 is already complete for every bucket.
        for bucket in 1:256
            ws.lookback[ONESWEEP_PASS_URS._lookback_index(0, bucket)] = ONESWEEP_PASS_URS._global_entry(UInt32(0))
        end

        # Pretend earlier tiles contributed three zeros and four ones.
        ws.lookback[ONESWEEP_PASS_URS._lookback_index(0, 1)] = ONESWEEP_PASS_URS._global_entry(UInt32(3))
        ws.lookback[ONESWEEP_PASS_URS._lookback_index(0, 2)] = ONESWEEP_PASS_URS._global_entry(UInt32(4))

        # Pass-wide bucket starts are CUB's d_bins equivalent.
        ws.bucket_offsets[ONESWEEP_PASS_URS._bucket_offsets_index(pass, 1)] = UInt32(1)
        ws.bucket_offsets[ONESWEEP_PASS_URS._bucket_offsets_index(pass, 2)] = UInt32(10)

        # Tile 1 local bins and exclusive_digit_prefix.
        ws.local_counts[onesweep_pass_worker_bucket(1)] = UInt32(2)
        ws.local_counts[onesweep_pass_worker_bucket(2)] = UInt32(1)
        ws.local_offsets[onesweep_pass_worker_bucket(1)] = UInt32(0)
        ws.local_offsets[onesweep_pass_worker_bucket(2)] = UInt32(2)

        ONESWEEP_PASS_URS._resolve_lookback_global_offsets!(
            ws.lookback,
            ws.bucket_offsets,
            ws.local_counts,
            ws.local_offsets,
            ws.global_offsets,
            1,
            Val(pass),
        )

        return (
            offsets = UInt32[
                ws.global_offsets[onesweep_pass_worker_bucket(1)],
                ws.global_offsets[onesweep_pass_worker_bucket(2)],
            ],
            entries = UInt32[
                ws.lookback[ONESWEEP_PASS_URS._lookback_index(1, 1)],
                ws.lookback[ONESWEEP_PASS_URS._lookback_index(1, 2)],
            ],
        )
    end

    @test looked_back.offsets == UInt32[4, 12]

    # Tile 1 now publishes complete same-bucket prefixes for following tiles.
    @test ONESWEEP_PASS_URS._is_global_entry(looked_back.entries[1])
    @test ONESWEEP_PASS_URS._is_global_entry(looked_back.entries[2])
    @test ONESWEEP_PASS_URS._entry_count(looked_back.entries[1]) == UInt32(5)
    @test ONESWEEP_PASS_URS._entry_count(looked_back.entries[2]) == UInt32(5)
end
