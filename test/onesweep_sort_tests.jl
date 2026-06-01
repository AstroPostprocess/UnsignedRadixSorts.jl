# ========================================================================== #
#  Test: Onesweep radix sort
# ========================================================================== #
#
#  What this file tests
#  1. Bucket-offset preparation
#     - prepare_bucket_offsets! computes 1-based exclusive bucket starts for
#       every byte pass and every supported unsigned key type.
#
#  2. Full public API
#     - onesweep_sort! sorts in-place and returns nothing for all supported
#       unsigned key types.
#     - onesweep_sortperm! sorts in-place and returns stable 1-based source
#       indices for the sorted keys.
#     - Small tile sizes exercise multi-tile lookback behavior.
#     - UInt8 exercises the odd-pass copy-back path.
#
#  3. Workspace API
#     - Explicit OnesweepWorkspace calls sort correctly and can be reused across
#       different inputs and lengths.
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

const ONESWEEP_URS = UnsignedRadixSorts
const ONESWEEP_UNSIGNED_TYPES = (UInt8, UInt16, UInt32, UInt64, UInt128)

function onesweep_digit_reference(x::T, pass::Int) where {T<:Unsigned}
    return UInt8((x >> (8 * (pass - 1))) & T(0xff))
end

function onesweep_histogram_reference(codes::Vector{T}, pass::Int) where {T<:Unsigned}
    counts = zeros(Int, 256)
    for x in codes
        counts[Int(onesweep_digit_reference(x, pass)) + 1] += 1
    end
    return counts
end

function onesweep_splitmix64_step(state::UInt64)
    state += 0x9e3779b97f4a7c15
    z = state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return state, z ⊻ (z >> 31)
end

function onesweep_deterministic_codes(::Type{T}, n::Int, seed::UInt64) where {T<:Unsigned}
    values = Vector{T}(undef, n)
    state = seed
    for i in 1:n
        state, lo = onesweep_splitmix64_step(state)
        if sizeof(T) < 8
            values[i] = T(lo & UInt64(typemax(T)))
        elseif sizeof(T) == 8
            values[i] = T(lo)
        else
            state, hi = onesweep_splitmix64_step(state)
            values[i] = T((UInt128(hi) << 64) | UInt128(lo))
        end
    end
    return values
end

function onesweep_duplicate_heavy_codes(::Type{T}, n::Int, seed::Int=0) where {T<:Unsigned}
    palette = T[
        zero(T),
        one(T),
        T(2),
        T(7),
        T(17),
        T(255),
        typemax(T),
        typemin(T),
    ]
    return [palette[mod(seed + 7 * i, length(palette)) + 1] for i in 1:n]
end

function onesweep_interleaved_duplicate_codes(::Type{T}, repeats::Int; key_count::Int=11) where {T<:Unsigned}
    values = Vector{T}(undef, repeats * key_count)
    index = 1
    for _ in 1:repeats
        for key in 0:(key_count - 1)
            values[index] = T(key)
            index += 1
        end
    end
    return values
end

function onesweep_extreme_mix(::Type{T}) where {T<:Unsigned}
    return T[
        typemax(T),
        zero(T),
        typemin(T),
        one(T),
        typemax(T) - one(T),
        T(3),
        zero(T),
        typemax(T),
    ]
end

function run_onesweep_sort_case(original::Vector{T}, tile_size::Int=4096) where {T<:Unsigned}
    codes = copy(original)
    result = onesweep_sort!(codes, Val(tile_size))

    @test result === nothing
    @test codes == sort(collect(original))
end

function run_onesweep_sort_workspace_case(original::Vector{T}, tile_size::Int=4096) where {T<:Unsigned}
    codes = copy(original)
    ws = OnesweepWorkspace(Vector{T})
    result = onesweep_sort!(codes, ws, Val(tile_size))

    @test result === nothing
    @test codes == sort(collect(original))
    @test length(ws.dst) == length(original)
    @test length(ws.bucket_offsets) == 256 * sizeof(T)
    @test length(ws.lookback) == 256 * cld(length(original), tile_size)
end

function run_onesweep_sortperm_case(original::Vector{T}, tile_size::Int=4096) where {T<:Unsigned}
    codes = copy(original)
    order = onesweep_sortperm!(codes, Val(tile_size))
    expected_order = sortperm(original, alg=Base.Sort.DEFAULT_STABLE)

    @test codes == sort(collect(original))
    @test collect(order) == expected_order
    @test codes == original[collect(Int, order)]
    @test eltype(order) == UInt32
end

function run_onesweep_sortperm_workspace_case(original::Vector{T}, tile_size::Int=4096) where {T<:Unsigned}
    codes = copy(original)
    ws = OnesweepWorkspace(Vector{T})
    order = onesweep_sortperm!(codes, ws, Val(tile_size))
    expected_order = sortperm(original, alg=Base.Sort.DEFAULT_STABLE)

    @test codes == sort(collect(original))
    @test collect(order) == expected_order
    @test codes == original[collect(Int, order)]
    @test eltype(order) == UInt32
    @test order === (isodd(sizeof(T)) ? ws.perms[2] : ws.perms[1])
    @test length(ws.perms[1]) == length(original)
    @test length(ws.perms[2]) == length(original)
    @test length(ws.bucket_offsets) == 256 * sizeof(T)
    @test length(ws.lookback) == 256 * cld(length(original), tile_size)
end

function assert_onesweep_bucket_offsets(codes::Vector{T}, tile_size::Int=7) where {T<:Unsigned}
    ws = OnesweepWorkspace(Vector{T})
    nelems = length(codes)
    ntiles = cld(nelems, tile_size)
    nworkers = Threads.nthreads()

    initialize_workspace!(ws, nelems, ntiles, Val(nworkers), Val(tile_size))
    prepare_bucket_offsets!(ws, codes)

    for pass in 1:sizeof(T)
        counts = onesweep_histogram_reference(codes, pass)
        running = UInt32(1)
        expected = Vector{UInt32}(undef, 256)

        for bucket in 1:256
            expected[bucket] = running
            running += UInt32(counts[bucket])
        end

        range = ONESWEEP_URS._bucket_offsets_index(pass, 1):ONESWEEP_URS._bucket_offsets_index(pass, 256)
        @test ws.bucket_offsets[range] == expected
        @test running == UInt32(length(codes) + 1)
    end
end

@testset "Onesweep bucket offsets" begin
    lengths = (0, 1, 2, 7, 31, 64)

    for T in ONESWEEP_UNSIGNED_TYPES
        @testset "$(T)" begin
            assert_onesweep_bucket_offsets(T[])
            assert_onesweep_bucket_offsets(onesweep_extreme_mix(T))

            for (case_id, n) in enumerate(lengths)
                assert_onesweep_bucket_offsets(
                    onesweep_deterministic_codes(T, n, UInt64(case_id) + 0x0ff5e7),
                )
                assert_onesweep_bucket_offsets(onesweep_duplicate_heavy_codes(T, n, case_id))
            end
        end
    end
end

@testset "Onesweep full sort" begin
    lengths = (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)

    for T in ONESWEEP_UNSIGNED_TYPES
        @testset "$(T)" begin
            run_onesweep_sort_case(T[])
            run_onesweep_sort_case(T[one(T)])
            run_onesweep_sort_case(T[zero(T), one(T)])
            run_onesweep_sort_case(T[one(T), zero(T)])
            run_onesweep_sort_case(fill(T(7), 19))
            run_onesweep_sort_case(onesweep_extreme_mix(T))
            run_onesweep_sort_case(onesweep_interleaved_duplicate_codes(T, 17))

            for (case_id, n) in enumerate(lengths)
                random_codes = onesweep_deterministic_codes(T, n, UInt64(case_id) + 0x5000)
                duplicate_heavy = onesweep_duplicate_heavy_codes(T, n, case_id)
                sorted_codes = sort(copy(random_codes))
                reverse_codes = reverse(sorted_codes)

                run_onesweep_sort_case(random_codes)
                run_onesweep_sort_case(duplicate_heavy)
                run_onesweep_sort_case(sorted_codes)
                run_onesweep_sort_case(reverse_codes)
            end
        end
    end
end

@testset "Onesweep small-tile sort" begin
    tile_sizes = (1, 2, 3, 7, 16)
    lengths = (0, 1, 2, 3, 7, 17, 33, 64)

    for T in ONESWEEP_UNSIGNED_TYPES
        @testset "$(T)" begin
            for tile_size in tile_sizes
                for (case_id, n) in enumerate(lengths)
                    random_codes = onesweep_deterministic_codes(
                        T,
                        n,
                        UInt64(case_id + 31 * tile_size) + 0x7000,
                    )
                    duplicate_heavy = onesweep_duplicate_heavy_codes(T, n, case_id + tile_size)

                    run_onesweep_sort_case(random_codes, tile_size)
                    run_onesweep_sort_case(duplicate_heavy, tile_size)
                end
            end
        end
    end
end

@testset "Onesweep sortperm / stability" begin
    lengths = (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)

    for T in ONESWEEP_UNSIGNED_TYPES
        @testset "$(T)" begin
            run_onesweep_sortperm_case(T[])
            run_onesweep_sortperm_case(T[one(T)])
            run_onesweep_sortperm_case(T[zero(T), one(T)])
            run_onesweep_sortperm_case(T[one(T), zero(T)])
            run_onesweep_sortperm_case(fill(T(7), 19))
            run_onesweep_sortperm_case(onesweep_extreme_mix(T))
            run_onesweep_sortperm_case(onesweep_interleaved_duplicate_codes(T, 17))

            for (case_id, n) in enumerate(lengths)
                random_codes = onesweep_deterministic_codes(T, n, UInt64(case_id) + 0xb000)
                duplicate_heavy = onesweep_duplicate_heavy_codes(T, n, case_id)
                sorted_codes = sort(copy(random_codes))
                reverse_codes = reverse(sorted_codes)

                run_onesweep_sortperm_case(random_codes)
                run_onesweep_sortperm_case(duplicate_heavy)
                run_onesweep_sortperm_case(sorted_codes)
                run_onesweep_sortperm_case(reverse_codes)
            end
        end
    end
end

@testset "Onesweep small-tile sortperm" begin
    tile_sizes = (1, 2, 3, 7, 16)
    lengths = (0, 1, 2, 3, 7, 17, 33, 64)

    for T in ONESWEEP_UNSIGNED_TYPES
        @testset "$(T)" begin
            for tile_size in tile_sizes
                for (case_id, n) in enumerate(lengths)
                    random_codes = onesweep_deterministic_codes(
                        T,
                        n,
                        UInt64(case_id + 31 * tile_size) + 0xc000,
                    )
                    duplicate_heavy = onesweep_duplicate_heavy_codes(T, n, case_id + tile_size)

                    run_onesweep_sortperm_case(random_codes, tile_size)
                    run_onesweep_sortperm_case(duplicate_heavy, tile_size)
                end
            end
        end
    end
end

@testset "Onesweep workspace sortperm" begin
    lengths = (0, 1, 2, 7, 31, 64)

    for T in ONESWEEP_UNSIGNED_TYPES
        @testset "$(T)" begin
            for (case_id, n) in enumerate(lengths)
                random_codes = onesweep_deterministic_codes(T, n, UInt64(case_id) + 0xd000)
                duplicate_heavy = onesweep_duplicate_heavy_codes(T, n, case_id)

                run_onesweep_sortperm_workspace_case(random_codes, 3)
                run_onesweep_sortperm_workspace_case(duplicate_heavy, 5)
                run_onesweep_sortperm_workspace_case(sort(copy(random_codes)), 7)
                run_onesweep_sortperm_workspace_case(reverse(sort(copy(random_codes))), 11)
            end

            ws = OnesweepWorkspace(Vector{T})
            first = onesweep_deterministic_codes(T, 64, UInt64(0xe110))
            second = onesweep_duplicate_heavy_codes(T, 19, 9)
            third = reverse(sort(copy(first)))

            codes = copy(first)
            order = onesweep_sortperm!(codes, ws, Val(3))
            @test collect(order) == sortperm(first, alg=Base.Sort.DEFAULT_STABLE)
            @test codes == first[collect(Int, order)]

            codes = copy(second)
            order = onesweep_sortperm!(codes, ws, Val(5))
            @test collect(order) == sortperm(second, alg=Base.Sort.DEFAULT_STABLE)
            @test codes == second[collect(Int, order)]

            codes = copy(third)
            order = onesweep_sortperm!(codes, ws, Val(7))
            @test collect(order) == sortperm(third, alg=Base.Sort.DEFAULT_STABLE)
            @test codes == third[collect(Int, order)]
        end
    end
end

@testset "Onesweep workspace sort" begin
    lengths = (0, 1, 2, 7, 31, 64)

    for T in ONESWEEP_UNSIGNED_TYPES
        @testset "$(T)" begin
            for (case_id, n) in enumerate(lengths)
                random_codes = onesweep_deterministic_codes(T, n, UInt64(case_id) + 0x9000)
                duplicate_heavy = onesweep_duplicate_heavy_codes(T, n, case_id)

                run_onesweep_sort_workspace_case(random_codes, 3)
                run_onesweep_sort_workspace_case(duplicate_heavy, 5)
                run_onesweep_sort_workspace_case(sort(copy(random_codes)), 7)
                run_onesweep_sort_workspace_case(reverse(sort(copy(random_codes))), 11)
            end

            ws = OnesweepWorkspace(Vector{T})
            first = onesweep_deterministic_codes(T, 64, UInt64(0xa110))
            second = onesweep_duplicate_heavy_codes(T, 19, 9)
            third = reverse(sort(copy(first)))

            codes = copy(first)
            @test onesweep_sort!(codes, ws, Val(3)) === nothing
            @test codes == sort(copy(first))

            codes = copy(second)
            @test onesweep_sort!(codes, ws, Val(5)) === nothing
            @test codes == sort(copy(second))

            codes = copy(third)
            @test onesweep_sort!(codes, ws, Val(7)) === nothing
            @test codes == sort(copy(third))
        end
    end
end
