# ========================================================================== #
#  Test: OneSweep pass LoadKeys
# ========================================================================== #
#
#  What this file tests
#  1. Tile loading
#     - `_load_keys!` copies the claimed source range into the worker's cached
#       key storage, matching CUB `LoadKeys`.
#
#  2. Temporary storage ownership
#     - The helper returns the same `keys` buffer and leaves entries outside
#       the valid tile range untouched.
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

@testset "OneSweep pass LoadKeys" begin
    tile_size = 8

    for T in (UInt8, UInt16, UInt32, UInt64, UInt128)
        @testset "$(T)" begin
            loaded = onesweep_on_default_thread() do
                src = T[0, 1, 2, 3, 4, 5, 6, 7, 8]
                ws = onesweep_pass_workspace(T, length(src), tile_size)
                tile_base = onesweep_pass_worker_base(tile_size)

                # Sentinel values make it clear that LoadKeys only writes tile_len keys.
                fill!(ws.keys, typemax(T))
                keys = ONESWEEP_PASS_URS._load_keys!(src, ws.keys, 3, 5, Val(tile_size))

                return (
                    same_buffer = keys === ws.keys,
                    loaded_keys = copy(ws.keys[(tile_base + 1):(tile_base + 5)]),
                    expected_keys = copy(src[3:7]),
                    tail_key = ws.keys[tile_base + 6],
                )
            end

            @test loaded.same_buffer
            @test loaded.loaded_keys == loaded.expected_keys
            @test loaded.tail_key == typemax(T)
        end
    end
end
