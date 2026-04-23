# ========================================================================== #
#  Test: Unsigned integer radix sort
# ========================================================================== #
#
#  What this file tests
#  1. Helper correctness
#     - _radix_digit returns the expected byte for known values.
#     - _radix_bucket is 1-based and matches the digit plus one.
#
#  2. Counting-sort building blocks
#     - radix_histogram! clears old state and counts every element exactly once.
#     - radix_offsets! computes 1-based exclusive bucket starts.
#     - radix_scatter! preserves the multiset and remains stable.
#
#  3. Single-pass behavior
#     - radix_pass! matches a stable counting-sort reference for one byte.
#     - Pathological distributions stress one-bucket and near-all-bucket cases.
#
#  4. Full public API
#     - radix_sort! sorts in-place and returns nothing.
#     - radix_sortperm! sorts in-place, returns a valid permutation, and is stable.
#
#  The cases are intentionally chosen to remain sensitive after future
#  parallelization. Repeated runs, interleaved duplicates, tiny off-by-one
#  inputs, and highly skewed bucket populations are all included.
#
# ========================================================================== #

using Test
using UnsignedIntegerRadixSort

const uirs = UnsignedIntegerRadixSort
const UNSIGNED_TYPES = (UInt8, UInt16, UInt32, UInt64, UInt128)

num_passes(::Type{T}) where {T<:Unsigned} = sizeof(T)

function digit_reference(x::T, pass::Int) where {T<:Unsigned}
    return UInt8((x >> (8 * (pass - 1))) & T(0xff))
end

function histogram_reference(codes::AbstractVector{T}, pass::Int) where {T<:Unsigned}
    counts = zeros(Int, 256)
    for x in codes
        counts[Int(digit_reference(x, pass)) + 1] += 1
    end
    return counts
end

function offsets_reference(counts::AbstractVector{Int})
    offsets = similar(counts)
    position = 1
    for i in eachindex(counts)
        offsets[i] = position
        position += counts[i]
    end
    return offsets
end

function stable_pass_reference(codes::AbstractVector{T}, pass::Int) where {T<:Unsigned}
    return sort(collect(codes); alg=MergeSort, by=x -> digit_reference(x, pass))
end

function stable_pass_reference(
    codes::AbstractVector{T},
    order::AbstractVector{Int},
    pass::Int,
) where {T<:Unsigned}
    perm = sort(
        collect(eachindex(codes));
        alg=MergeSort,
        by=i -> digit_reference(codes[i], pass),
    )
    return codes[perm], order[perm]
end

function stable_sortperm_reference(codes::AbstractVector{T}) where {T<:Unsigned}
    return sort(collect(eachindex(codes)); alg=MergeSort, by=i -> codes[i])
end

function splitmix64_step(state::UInt64)
    state += 0x9e3779b97f4a7c15
    z = state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return state, z ⊻ (z >> 31)
end

function deterministic_codes(::Type{T}, n::Int, seed::UInt64) where {T<:Unsigned}
    values = Vector{T}(undef, n)
    state = seed
    for i in 1:n
        state, lo = splitmix64_step(state)
        if sizeof(T) < 8
            values[i] = T(lo & UInt64(typemax(T)))
        elseif sizeof(T) == 8
            values[i] = T(lo)
        else
            state, hi = splitmix64_step(state)
            values[i] = (UInt128(hi) << 64) | UInt128(lo)
        end
    end
    return values
end

function assert_valid_permutation(order::AbstractVector{Int}, n::Int)
    @test length(order) == n
    @test sort(collect(order)) == collect(1:n)
end

function assert_stable_equal_keys(original::AbstractVector{T}, order::AbstractVector{Int}) where {T<:Unsigned}
    original_positions = Dict{T, Vector{Int}}()
    sorted_positions = Dict{T, Vector{Int}}()

    for i in eachindex(original)
        push!(get!(original_positions, original[i], Int[]), i)
    end
    for idx in order
        push!(get!(sorted_positions, original[idx], Int[]), idx)
    end

    @test length(sorted_positions) == length(original_positions)
    for key in keys(original_positions)
        @test haskey(sorted_positions, key)
        @test sorted_positions[key] == original_positions[key]
    end
end

function make_same_bucket_codes(::Type{T}, pass::Int, n::Int; digit::UInt8=0x5a) where {T<:Unsigned}
    shift = 8 * (pass - 1)
    bucket_mask = T(0xff) << shift
    values = Vector{T}(undef, n)
    for i in 1:n
        seed = T(i - 1) * T(0x0101)
        values[i] = (seed & ~bucket_mask) | (T(digit) << shift)
    end
    return values
end

function make_dense_bucket_codes(::Type{T}, pass::Int; rounds::Int=2) where {T<:Unsigned}
    shift = 8 * (pass - 1)
    bucket_mask = T(0xff) << shift
    values = Vector{T}(undef, 256 * rounds)
    index = 1
    for round in 0:(rounds - 1)
        for digit in 0:255
            seed = T(round + 1) * T(0x1111) + T(255 - digit)
            values[index] = (seed & ~bucket_mask) | (T(digit) << shift)
            index += 1
        end
    end
    return values
end

function make_interleaved_duplicate_codes(::Type{T}, repeats::Int; key_count::Int=11) where {T<:Unsigned}
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

function make_duplicate_heavy_codes(::Type{T}, n::Int, seed::Int=0) where {T<:Unsigned}
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

function make_extreme_mix(::Type{T}) where {T<:Unsigned}
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

function run_radix_sort_case(original::AbstractVector{T}) where {T<:Unsigned}
    codes = copy(original)
    result = radix_sort!(codes)
    @test result === nothing
    @test codes == sort(collect(original))
end

function run_radix_sort_workspace_case(original::AbstractVector{T}) where {T<:Unsigned}
    codes = copy(original)
    ws = RadixSortWorkspace(T, length(codes))
    result = radix_sort!(codes, ws)
    @test result === nothing
    @test codes == sort(collect(original))
end

function run_radix_sortperm_case(original::AbstractVector{T}) where {T<:Unsigned}
    codes = copy(original)
    order = radix_sortperm!(codes)
    reference_order = stable_sortperm_reference(original)

    @test order isa Vector{Int}
    @test codes == sort(collect(original))
    @test codes == original[order]
    assert_valid_permutation(order, length(original))
    assert_stable_equal_keys(original, order)
    @test order == reference_order
end

function run_radix_sortperm_workspace_case(original::AbstractVector{T}) where {T<:Unsigned}
    codes = copy(original)
    ws = RadixSortPermWorkspace(T, length(codes))
    order = radix_sortperm!(codes, ws)
    reference_order = stable_sortperm_reference(original)

    @test order === ws.order
    @test order isa Vector{Int}
    @test codes == sort(collect(original))
    @test codes == original[order]
    assert_valid_permutation(order, length(original))
    assert_stable_equal_keys(original, order)
    @test order == reference_order
end

function radix_sort_workspace_allocations(::Type{T}, n::Int) where {T<:Unsigned}
    original = deterministic_codes(T, n, UInt64(0x12345678))
    codes = copy(original)
    ws = RadixSortWorkspace(T, n)
    radix_sort!(codes, ws)
    copyto!(codes, original)
    return @allocated radix_sort!(codes, ws)
end

function radix_sortperm_workspace_allocations(::Type{T}, n::Int) where {T<:Unsigned}
    original = deterministic_codes(T, n, UInt64(0x87654321))
    codes = copy(original)
    ws = RadixSortPermWorkspace(T, n)
    radix_sortperm!(codes, ws)
    copyto!(codes, original)
    return @allocated radix_sortperm!(codes, ws)
end

@testset "Radix helpers" begin
    @testset "_radix_digit" begin
        x = UInt32(0x1234abcd)
        @test uirs._radix_digit(x, Val(1)) == 0xcd
        @test uirs._radix_digit(x, Val(2)) == 0xab
        @test uirs._radix_digit(x, Val(3)) == 0x34
        @test uirs._radix_digit(x, Val(4)) == 0x12

        y = (UInt128(0x0123456789abcdef) << 64) | UInt128(0xfedcba9876543210)
        @test uirs._radix_digit(y, Val(1)) == 0x10
        @test uirs._radix_digit(y, Val(8)) == 0xfe
        @test uirs._radix_digit(y, Val(9)) == 0xef
        @test uirs._radix_digit(y, Val(16)) == 0x01
    end

    @testset "_radix_bucket" begin
        @test uirs._radix_bucket(UInt8(0x00), Val(1)) == 1
        @test uirs._radix_bucket(UInt8(0xff), Val(1)) == 256
        @test uirs._radix_bucket(UInt32(0x1234abcd), Val(1)) == 0xcd + 1
        @test uirs._radix_bucket(UInt32(0x1234abcd), Val(2)) == 0xab + 1

        for T in UNSIGNED_TYPES
            @test uirs._radix_bucket(typemin(T), Val(1)) == 1
            @test uirs._radix_bucket(typemax(T), Val(1)) == 256
        end
    end
end

@testset "Histogram / offsets / scatter" begin
    @testset "radix_histogram! clears old state and counts exactly" begin
        counts = fill(-1, 256)
        codes = UInt16[0x0000, 0x00ff, 0x1200, 0x12ff, 0x1200, 0x34ab]

        radix_histogram!(counts, codes, Val(1))
        @test counts == histogram_reference(codes, 1)
        @test sum(counts) == length(codes)
        @test counts[1] == 3
        @test counts[256] == 2
        @test counts[0xab + 1] == 1

        fill!(counts, -7)
        radix_histogram!(counts, UInt16[], Val(2))
        @test counts == zeros(Int, 256)
        @test sum(counts) == 0

        dense = make_dense_bucket_codes(UInt32, 1; rounds=2)
        fill!(counts, 99)
        radix_histogram!(counts, dense, Val(1))
        @test counts == histogram_reference(dense, 1)
        @test sum(counts) == length(dense)
        @test all(==(2), counts)
    end

    @testset "radix_offsets! computes 1-based exclusive starts" begin
        counts = [0, 2, 0, 1, 3, 0]
        offsets = fill(-1, length(counts))

        radix_offsets!(offsets, counts)
        @test offsets == [1, 1, 3, 3, 4, 7]

        counts_256 = zeros(Int, 256)
        counts_256[1] = 2
        counts_256[2] = 1
        counts_256[255] = 3
        counts_256[256] = 1
        offsets_256 = fill(-1, 256)

        radix_offsets!(offsets_256, counts_256)
        @test offsets_256 == offsets_reference(counts_256)
        @test offsets_256[1] == 1
        @test offsets_256[2] == 3
        @test offsets_256[3] == 4
        @test offsets_256[255] == 4
        @test offsets_256[256] == 7
    end

    @testset "radix_scatter! is stable and preserves the multiset" begin
        codes = UInt16[0x0102, 0x0201, 0x0302, 0x0401, 0x0502, 0x0601]
        counts = histogram_reference(codes, 1)
        offsets = offsets_reference(counts)
        out = similar(codes)
        original_offsets = copy(offsets)

        radix_scatter!(out, offsets, codes, Val(1))

        @test out == stable_pass_reference(codes, 1)
        @test sort(collect(out)) == sort(collect(codes))
        @test out[1:3] == UInt16[0x0201, 0x0401, 0x0601]
        @test out[4:6] == UInt16[0x0102, 0x0302, 0x0502]
        @test offsets == original_offsets .+ counts
    end

    @testset "radix_scatter! with permutation output stays aligned and stable" begin
        codes = UInt16[0x0102, 0x0201, 0x0302, 0x0401, 0x0502, 0x0601]
        order = collect(eachindex(codes))
        counts = histogram_reference(codes, 1)
        offsets = offsets_reference(counts)
        out_codes = similar(codes)
        out_order = similar(order)

        radix_scatter!(out_codes, out_order, offsets, codes, order, Val(1))

        expected_codes, expected_order = stable_pass_reference(codes, order, 1)
        @test out_codes == expected_codes
        @test out_order == expected_order
        @test out_codes == codes[out_order]
        @test out_order[1:3] == [2, 4, 6]
        @test out_order[4:6] == [1, 3, 5]
    end
end

@testset "Single-pass behavior" begin
    @testset "radix_pass! matches stable counting sort for one byte" begin
        codes = UInt32[
            0x1234ab10,
            0x0000ab20,
            0xffffab10,
            0x0102cd30,
            0x9999ab20,
            0x4242cd10,
        ]
        counts = fill(-1, 256)
        offsets = fill(-1, 256)
        out = similar(codes)

        radix_pass!(out, counts, offsets, codes, Val(2))

        @test out == stable_pass_reference(codes, 2)
        @test counts == histogram_reference(codes, 2)
        @test offsets == offsets_reference(counts) .+ counts
    end

    @testset "radix_pass! with order matches stable single-byte reference" begin
        codes = UInt32[
            0x0100007f,
            0x02000011,
            0x0300007f,
            0x04000022,
            0x0500007f,
            0x06000011,
        ]
        order = collect(eachindex(codes))
        counts = fill(-1, 256)
        offsets = fill(-1, 256)
        out_codes = similar(codes)
        out_order = similar(order)

        radix_pass!(out_codes, out_order, counts, offsets, codes, order, Val(1))

        expected_codes, expected_order = stable_pass_reference(codes, order, 1)
        @test out_codes == expected_codes
        @test out_order == expected_order
        @test out_codes == codes[out_order]
    end

    @testset "single-bucket inputs keep their original order" begin
        for repeat in 1:12
            codes = make_same_bucket_codes(UInt32, 1, 257; digit=0x42)
            counts = fill(-1, 256)
            offsets = fill(-1, 256)
            out = similar(codes)

            radix_pass!(out, counts, offsets, codes, Val(1))

            @test out == codes
            @test counts[0x42 + 1] == length(codes)
            @test sum(counts) == length(codes)
        end
    end

    @testset "dense bucket occupancy exercises every bucket" begin
        codes = make_dense_bucket_codes(UInt32, 1; rounds=2)
        counts = fill(-1, 256)
        offsets = fill(-1, 256)
        out = similar(codes)

        radix_pass!(out, counts, offsets, codes, Val(1))

        @test counts == fill(2, 256)
        @test out == stable_pass_reference(codes, 1)
    end

    @testset "tiny off-by-one-sensitive pass cases" begin
        cases = (
            UInt16[],
            UInt16[0x0000],
            UInt16[0x0001, 0x0000],
            UInt16[0x00ff, 0x0000, 0x00ff],
            UInt16[0x0000, 0x00ff, 0x0001, 0x0000],
        )

        for codes in cases
            counts = fill(-1, 256)
            offsets = fill(-1, 256)
            out = similar(codes)
            radix_pass!(out, counts, offsets, codes, Val(1))
            @test out == stable_pass_reference(codes, 1)
            @test counts == histogram_reference(codes, 1)
            @test sum(counts) == length(codes)
        end
    end
end

@testset "Workspace constructors" begin
    for T in UNSIGNED_TYPES
        @testset "$(T)" begin
            ws = RadixSortWorkspace(T, 17)
            @test ws isa RadixSortWorkspace{T}
            @test length(ws.tmp) == 17
            @test length(ws.counts) == 256
            @test length(ws.offsets) == 256

            perm_ws = RadixSortPermWorkspace(T, 17)
            @test perm_ws isa RadixSortPermWorkspace{T}
            @test length(perm_ws.tmp_codes) == 17
            @test length(perm_ws.order) == 17
            @test length(perm_ws.tmp_order) == 17
            @test length(perm_ws.counts) == 256
            @test length(perm_ws.offsets) == 256
        end
    end

    @test_throws ArgumentError RadixSortWorkspace(UInt32, -1)
    @test_throws ArgumentError RadixSortPermWorkspace(UInt32, -1)
end

@testset "Full sort" begin
    base_lengths = (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)

    for T in UNSIGNED_TYPES
        @testset "$(T)" begin
            run_radix_sort_case(T[])
            run_radix_sort_case(T[one(T)])
            run_radix_sort_case(T[zero(T), one(T)])
            run_radix_sort_case(T[one(T), zero(T)])
            run_radix_sort_case(fill(T(7), 19))
            run_radix_sort_case(T[zero(T), typemax(T), one(T), typemax(T), typemin(T), T(3)])
            run_radix_sort_case(make_extreme_mix(T))
            run_radix_sort_case(make_interleaved_duplicate_codes(T, 23))

            for (case_id, n) in enumerate(base_lengths)
                random_codes = deterministic_codes(T, n, UInt64(case_id) + 0x0000000000005eed)
                duplicate_heavy = make_duplicate_heavy_codes(T, n, case_id)
                sorted_codes = sort(copy(random_codes))
                reverse_codes = reverse(sorted_codes)

                run_radix_sort_case(random_codes)
                run_radix_sort_case(duplicate_heavy)
                run_radix_sort_case(sorted_codes)
                run_radix_sort_case(reverse_codes)
            end

            for repeat in 1:10
                interleaved = make_interleaved_duplicate_codes(T, 29)
                run_radix_sort_case(interleaved)
            end
        end
    end
end

@testset "Workspace sort" begin
    lengths = (0, 1, 2, 7, 31, 128)

    for T in UNSIGNED_TYPES
        @testset "$(T)" begin
            for (case_id, n) in enumerate(lengths)
                random_codes = deterministic_codes(T, n, UInt64(case_id) + 0x1000)
                duplicate_heavy = make_duplicate_heavy_codes(T, n, case_id)
                interleaved = n == 0 ? T[] : resize!(copy(make_interleaved_duplicate_codes(T, max(1, cld(n, 11)))), n)

                run_radix_sort_workspace_case(random_codes)
                run_radix_sort_workspace_case(duplicate_heavy)
                run_radix_sort_workspace_case(sort(copy(random_codes)))
                run_radix_sort_workspace_case(reverse(sort(copy(random_codes))))
                run_radix_sort_workspace_case(interleaved)
            end

            n = 64
            ws = RadixSortWorkspace(T, n)
            first = deterministic_codes(T, n, UInt64(0xabc0))
            second = make_duplicate_heavy_codes(T, n, 9)
            third = reverse(sort(copy(first)))

            codes = copy(first)
            @test radix_sort!(codes, ws) === nothing
            @test codes == sort(copy(first))

            copyto!(codes, second)
            @test radix_sort!(codes, ws) === nothing
            @test codes == sort(copy(second))

            copyto!(codes, third)
            @test radix_sort!(codes, ws) === nothing
            @test codes == sort(copy(third))

            @test_throws DimensionMismatch radix_sort!(copy(first), RadixSortWorkspace(T, n + 1))
        end
    end
end

@testset "Sortperm / stability" begin
    base_lengths = (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)

    for T in UNSIGNED_TYPES
        @testset "$(T)" begin
            run_radix_sortperm_case(T[])
            run_radix_sortperm_case(T[one(T)])
            run_radix_sortperm_case(T[zero(T), one(T)])
            run_radix_sortperm_case(T[one(T), zero(T)])
            run_radix_sortperm_case(fill(T(7), 19))
            run_radix_sortperm_case(make_extreme_mix(T))
            run_radix_sortperm_case(make_interleaved_duplicate_codes(T, 31))
            run_radix_sortperm_case(T[typemax(T), zero(T), typemax(T), one(T), zero(T), typemin(T)])

            for (case_id, n) in enumerate(base_lengths)
                random_codes = deterministic_codes(T, n, UInt64(case_id) + 0x000000000000600d)
                duplicate_heavy = make_duplicate_heavy_codes(T, n, case_id)
                sorted_codes = sort(copy(random_codes))
                reverse_codes = reverse(sorted_codes)

                run_radix_sortperm_case(random_codes)
                run_radix_sortperm_case(duplicate_heavy)
                run_radix_sortperm_case(sorted_codes)
                run_radix_sortperm_case(reverse_codes)
            end

            for repeat in 1:12
                interleaved = make_interleaved_duplicate_codes(T, 37)
                run_radix_sortperm_case(interleaved)
            end
        end
    end
end

@testset "Workspace sortperm / stability" begin
    lengths = (0, 1, 2, 7, 31, 128)

    for T in UNSIGNED_TYPES
        @testset "$(T)" begin
            for (case_id, n) in enumerate(lengths)
                random_codes = deterministic_codes(T, n, UInt64(case_id) + 0x2000)
                duplicate_heavy = make_duplicate_heavy_codes(T, n, case_id)
                interleaved = n == 0 ? T[] : resize!(copy(make_interleaved_duplicate_codes(T, max(1, cld(n, 11)))), n)

                run_radix_sortperm_workspace_case(random_codes)
                run_radix_sortperm_workspace_case(duplicate_heavy)
                run_radix_sortperm_workspace_case(sort(copy(random_codes)))
                run_radix_sortperm_workspace_case(reverse(sort(copy(random_codes))))
                run_radix_sortperm_workspace_case(interleaved)
            end

            n = 64
            ws = RadixSortPermWorkspace(T, n)
            first = deterministic_codes(T, n, UInt64(0xdef0))
            second = make_duplicate_heavy_codes(T, n, 11)
            third = reverse(sort(copy(first)))

            codes = copy(first)
            order = radix_sortperm!(codes, ws)
            @test order === ws.order
            @test codes == sort(copy(first))
            @test codes == first[order]

            copyto!(codes, second)
            order = radix_sortperm!(codes, ws)
            @test order === ws.order
            @test codes == sort(copy(second))
            @test codes == second[order]

            copyto!(codes, third)
            order = radix_sortperm!(codes, ws)
            @test order === ws.order
            @test codes == sort(copy(third))
            @test codes == third[order]

            @test_throws DimensionMismatch radix_sortperm!(copy(first), RadixSortPermWorkspace(T, n + 1))
        end
    end
end

@testset "Workspace allocation behavior" begin
    n = 257
    for T in UNSIGNED_TYPES
        @testset "$(T)" begin
            @test radix_sort_workspace_allocations(T, n) == 0
            @test radix_sortperm_workspace_allocations(T, n) == 0
        end
    end
end
