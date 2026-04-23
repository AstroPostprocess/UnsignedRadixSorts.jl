using BenchmarkTools
using UnsignedIntegerRadixSort

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

function benchmark_radix_sort(::Type{T}, n::Int; seed::UInt64=0x5eed) where {T<:Unsigned}
    x = deterministic_codes(T, n, seed)
    y_workspace = similar(x)
    ws = RadixSortWorkspace(T, n)

    println()
    println("== $(T), n = $(n) ==")

    sort_trial = @benchmark sort!(y) setup=(y = copy($x)) evals=1
    radix_trial = @benchmark radix_sort!(y) setup=(y = copy($x)) evals=1
    workspace_trial = @benchmark radix_sort!($y_workspace, $ws) setup=(copyto!($y_workspace, $x)) evals=1

    println("sort!")
    display(sort_trial)
    println("radix_sort! (high-level)")
    display(radix_trial)
    println("radix_sort! (workspace)")
    display(workspace_trial)

    return (; sort_trial, radix_trial, workspace_trial)
end

function main()
    for T in (UInt32, UInt64, UInt128), n in (1024, 65536)
        benchmark_radix_sort(T, n)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
