"""
    radix_histogram!(counts :: VI, codes :: VU, v :: Val{N}) where {T <: Unsigned, N, VI <: AbstractVector{Int}, VU <: AbstractVector{T}}

Compute the 8-bit radix histogram for pass `N` from `codes` and write the result in-place to `counts`.

The function first resets `counts` to zero, then scans `codes` and increments the bucket corresponding to the `N`-th 8-bit digit of each unsigned value. Buckets follow Julia's 1-based indexing convention, so digit `0x00` maps to `counts[1]` and digit `0xff` maps to `counts[256]`.

# Parameters

- `counts :: VI`: Preallocated histogram buffer. It is overwritten in-place and is expected to have length `256`.
- `codes :: VU`: Input vector of unsigned radix keys.
- `v :: Val{N}`: Compile-time pass selector indicating which 8-bit digit to histogram. `Val(1)` selects the least significant byte, `Val(2)` the next byte, and so on.
"""
@inline function radix_histogram!(counts :: MI, codes :: VU, v :: Val{N}) where {N, T<:Unsigned, TI<:Integer, MI<:AbstractMatrix{TI}, VU<:AbstractVector{T}}
    # Get the number of chunks (threads)
    nchunks = size(counts, 2)
    # Get the total number of elements to histogram
    n = length(codes)
    # Initialize the counts to zero before accumulation.
    fill!(counts, zero(TI))

    # Parallel histogram accumulation across chunks
    Threads.@threads for cid in 1:nchunks
        lo, hi = _chunk_bounds(n, nchunks, cid)

        @inbounds for i in lo:hi
            bucket = _radix_bucket(codes[i], v)
            counts[bucket, cid] += one(TI)
        end
    end

    return nothing
end


# Toolbox
@inline function _chunk_bounds(n :: Integer, nchunks :: Integer, cid :: Integer)
    lo = div((cid - 1) * n, nchunks) + 1
    hi = div(cid * n, nchunks)
    return lo, hi
end
