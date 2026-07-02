# ========================================================================== #
#  Test: OneSweep pass Process() dataflow
# ========================================================================== #
#
#  What this file tests
#  1. Key-only Process()
#     - One CPU pass executes the same stage order as CUB Process():
#       LoadKeys -> RankKeys -> ScatterKeysShared -> LookbackGlobal ->
#       ScatterKeysGlobal.
#
#  2. Key/value Process()
#     - The permutation pass follows the same key dataflow and additionally
#       runs GatherScatterValues for 1-based source indices.
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

@testset "OneSweep pass Process dataflow" begin
    T = UInt16
    tile_size = 3
    pass = 1
    codes = T[0x0202, 0x0101, 0x0201, 0x0000, 0x0102, 0x0001, 0x0300]
    order = onesweep_pass_digit_order(codes, pass)
    # Reference for one radix pass: stable sort by the selected byte only.
    expected = codes[order]

    ws = onesweep_pass_workspace(T, length(codes), tile_size)
    # Process() assumes d_bins and d_lookback have already been prepared/reset.
    prepare_bucket_offsets!(ws, codes)
    reset_pass_workspace!(ws)

    output = onesweep_on_default_thread() do
        ONESWEEP_PASS_URS.onesweep_pass_kernel!(codes, ws, Val(tile_size), Val(pass))
        return copy(ws.dst)
    end

    @test output == expected
end

@testset "OneSweep perm pass Process dataflow" begin
    T = UInt16
    tile_size = 3
    pass = 1
    codes = T[0x0202, 0x0101, 0x0201, 0x0000, 0x0102, 0x0001, 0x0300]
    order = onesweep_pass_digit_order(codes, pass)
    # Keys and permutation values must follow the same stable digit order.
    expected = codes[order]

    ws = onesweep_perm_pass_workspace(T, length(codes), tile_size)
    # Prepare pass-wide bucket starts and clear the per-pass lookback state.
    prepare_bucket_offsets!(ws, codes)
    reset_pass_workspace!(ws)

    output = onesweep_on_default_thread() do
        ONESWEEP_PASS_URS.onesweep_perm_pass_kernel!(codes, ws, Val(tile_size), Val(pass))
        return (keys = copy(ws.dst), perm = copy(ws.perms[2]))
    end

    @test output.keys == expected
    @test output.perm == UInt32.(order)
end
