# ========================================================================== #
#  Test: OneSweep pass Scatter and GatherScatterValues
# ========================================================================== #
#
#  What this file tests
#  1. ScatterKeysShared
#     - Keys are staged into `s.keys_out` order using the CUB-style local ranks.
#
#  2. ScatterKeysGlobal
#     - Staged keys are written to global bucket positions from lookback
#       resolved `global_offsets`.
#
#  3. GatherScatterValues
#     - Permutation values follow the same ranks and final key positions.
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

@testset "OneSweep pass ScatterKeysShared and ScatterKeysGlobal" begin
    T = UInt16
    tile_size = 8
    pass = 1
    src = T[0x02, 0x01, 0x02, 0x00, 0x01, 0x02]
    ranks = UInt32[3, 1, 4, 0, 2, 5]

    scattered = onesweep_on_default_thread() do
        ws = onesweep_pass_workspace(T, length(src), tile_size)
        tile_base = onesweep_pass_worker_base(tile_size)

        ws.keys[(tile_base + 1):(tile_base + length(src))] = src
        ws.local_ranks[(tile_base + 1):(tile_base + length(src))] = ranks

        # ScatterKeysShared produces digit-sorted tile order in keys_out.
        ONESWEEP_PASS_URS._scatter_keys_shared!(ws.keys_out, ws.keys, ws.local_ranks, length(src), Val(tile_size))

        staged = copy(ws.keys_out[(tile_base + 1):(tile_base + length(src))])

        # LookbackGlobal would normally write these bucket bases.
        ws.global_offsets[onesweep_pass_worker_bucket(1)] = UInt32(10)
        ws.global_offsets[onesweep_pass_worker_bucket(2)] = UInt32(19)
        ws.global_offsets[onesweep_pass_worker_bucket(3)] = UInt32(27)

        # ScatterKeysGlobal consumes keys_out in sorted tile order.
        dst = fill(typemax(T), 40)
        ONESWEEP_PASS_URS._scatter_keys_global!(ws.keys_out, dst, ws.global_offsets, length(src), Val(tile_size), Val(pass))

        return (staged = staged, dst = dst)
    end

    @test scattered.staged == T[0x00, 0x01, 0x01, 0x02, 0x02, 0x02]
    @test scattered.dst[10] == T(0x00)
    @test scattered.dst[20:21] == T[0x01, 0x01]
    @test scattered.dst[30:32] == T[0x02, 0x02, 0x02]
end

@testset "OneSweep pass GatherScatterValues" begin
    T = UInt16
    tile_size = 8
    pass = 1
    src = T[0x02, 0x01, 0x02, 0x00, 0x01, 0x02]
    ranks = UInt32[3, 1, 4, 0, 2, 5]

    gathered = onesweep_on_default_thread() do
        ws = onesweep_pass_workspace(T, length(src), tile_size)
        tile_base = onesweep_pass_worker_base(tile_size)

        ws.keys[(tile_base + 1):(tile_base + length(src))] = src
        ws.local_ranks[(tile_base + 1):(tile_base + length(src))] = ranks
        ONESWEEP_PASS_URS._scatter_keys_shared!(ws.keys_out, ws.keys, ws.local_ranks, length(src), Val(tile_size))

        # Use sparse global bases so each bucket's output range is easy to inspect.
        ws.global_offsets[onesweep_pass_worker_bucket(1)] = UInt32(10)
        ws.global_offsets[onesweep_pass_worker_bucket(2)] = UInt32(19)
        ws.global_offsets[onesweep_pass_worker_bucket(3)] = UInt32(27)

        dst = fill(typemax(T), 40)
        perm_src = UInt32[0, 10, 20, 30, 40, 50, 60, 0]
        perm_dst = fill(UInt32(0), 40)

        # GatherScatterValues keeps permutation values aligned with sorted keys.
        ONESWEEP_PASS_URS._scatter_key_values_global!(
            ws.keys_out,
            ws.values_out,
            ws.keys,
            ws.local_ranks,
            dst,
            perm_src,
            perm_dst,
            ws.global_offsets,
            2,
            length(src),
            Val(tile_size),
            Val(pass),
        )

        return (dst = dst, perm_dst = perm_dst)
    end

    @test gathered.dst[10] == T(0x00)
    @test gathered.dst[20:21] == T[0x01, 0x01]
    @test gathered.dst[30:32] == T[0x02, 0x02, 0x02]
    @test gathered.perm_dst[10] == UInt32(40)
    @test gathered.perm_dst[20:21] == UInt32[20, 50]
    @test gathered.perm_dst[30:32] == UInt32[10, 30, 60]
end
