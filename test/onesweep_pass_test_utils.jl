# ========================================================================== #
#  Test helpers: OneSweep pass stages
# ========================================================================== #
#
#  What this file provides
#  1. Internal-helper access
#     - Tests call CPU OneSweep stage helpers through the package namespace.
#
#  2. Worker-context helpers
#     - Stage helpers use the default thread-pool worker id, so direct helper
#       tests run their body on a default worker.
#
#  3. Reference helpers
#     - Small workspace builders and stable digit-order references keep the
#       per-stage tests focused on one CUB Process() step at a time.
#
# ========================================================================== #

const ONESWEEP_PASS_URS = UnsignedRadixSorts

function onesweep_on_default_thread(f)
    # Run direct stage-helper calls on a default worker, matching the context
    # used by the threaded CPU pass.
    result = Ref{Any}()
    captured_error = Ref{Any}(nothing)

    Threads.@threads :static for _ in 1:1
        try
            result[] = f()
        catch err
            captured_error[] = err
        end
    end

    captured_error[] === nothing || throw(captured_error[])
    return result[]
end

function onesweep_pass_worker_base(tile_size::Int)
    # Per-worker tile temporary storage starts at this flat-buffer offset.
    return tile_size * (ONESWEEP_PASS_URS._worker_id() - 1)
end

function onesweep_pass_workspace(::Type{T}, nelems::Int, tile_size::Int) where {T<:Unsigned}
    # Allocate the key-only workspace slices used by one Process() stage.
    ws = OnesweepWorkspace(Vector{T})
    ntiles = cld(nelems, tile_size)
    initialize_workspace!(ws, nelems, ntiles, Val(Threads.nthreads()), Val(tile_size))
    return ws
end

function onesweep_perm_pass_workspace(::Type{T}, nelems::Int, tile_size::Int) where {T<:Unsigned}
    # Allocate the key/value workspace slices used by permutation passes.
    ws = OnesweepWorkspace(Vector{T})
    ntiles = cld(nelems, tile_size)
    initialize_perm_workspace!(ws, nelems, ntiles, Val(Threads.nthreads()), Val(tile_size))
    return ws
end

function onesweep_pass_digit_order(codes::Vector{T}, pass::Int) where {T<:Unsigned}
    # Stable reference order for one 8-bit radix pass.
    indices = collect(eachindex(codes))
    return sortperm(indices; by=i -> ONESWEEP_PASS_URS._radix_bucket(codes[i], pass), alg=Base.Sort.DEFAULT_STABLE)
end

function onesweep_pass_worker_bucket(bucket::Int)
    # Current worker's flattened 256-bucket slot.
    return ONESWEEP_PASS_URS._worker_bucket_index(ONESWEEP_PASS_URS._worker_id(), bucket)
end
