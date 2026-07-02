# ========================================================================== #
#  Test: OneSweep workspace temporary storage
# ========================================================================== #
#
#  What this file tests
#  1. CUB Process() temporary storage
#     - CPU workspace initialization allocates per-worker `keys`, `keys_out`,
#       and `values_out` slices that mirror CUB agent storage.
#
#  2. Lookback payload guard
#     - OneSweep lookback entries reserve two high bits for state, so the
#       lower 30-bit payload must be able to hold every prefix count.
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

@testset "Onesweep workspace CUB temporary storage" begin
    T = UInt32
    nelems = 17
    tile_size = 5
    ntiles = cld(nelems, tile_size)
    nworkers = Threads.nthreads()
    ws = OnesweepWorkspace(Vector{T})

    initialize_base_workspace!(ws, nelems, ntiles)

    # Base initialization only prepares pass-global storage.
    @test isempty(ws.keys)
    @test isempty(ws.keys_out)
    @test isempty(ws.values_out)

    initialize_workspace!(ws, nelems, ntiles, Val(nworkers), Val(tile_size))

    # Full CPU initialization adds the per-worker Process() staging buffers.
    @test length(ws.keys) == tile_size * nworkers
    @test length(ws.keys_out) == tile_size * nworkers
    @test length(ws.values_out) == tile_size * nworkers
    @test all(==(zero(T)), ws.keys)
    @test all(==(zero(T)), ws.keys_out)
    @test all(==(zero(UInt32)), ws.values_out)
end

@testset "OneSweep lookback payload limit" begin
    ws = OnesweepWorkspace(Vector{UInt32})
    limit = Int(typemax(UInt32) >> 2)

    # Reject sizes that cannot fit in the packed 30-bit lookback payload.
    @test_throws ArgumentError initialize_base_workspace!(ws, limit + 1, 1)
    @test isempty(ws.dst)
    @test isempty(ws.lookback)
end
