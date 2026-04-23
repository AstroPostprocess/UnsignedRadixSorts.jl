"""
    radix_histogram!(counts :: VI, codes :: VU, v :: Val{N}) where {T <: Unsigned, N, VI <: AbstractVector{Int}, VU <: AbstractVector{T}}

Compute the 8-bit radix histogram for pass `N` from `codes` and write the result in-place to `counts`.

The function first resets `counts` to zero, then scans `codes` and increments the bucket corresponding to the `N`-th 8-bit digit of each unsigned value. Buckets follow Julia's 1-based indexing convention, so digit `0x00` maps to `counts[1]` and digit `0xff` maps to `counts[256]`.

# Parameters

- `counts :: VI`: Preallocated histogram buffer. It is overwritten in-place and is expected to have length `256`.
- `codes :: VU`: Input vector of unsigned radix keys.
- `v :: Val{N}`: Compile-time pass selector indicating which 8-bit digit to histogram. `Val(1)` selects the least significant byte, `Val(2)` the next byte, and so on.
"""
@inline function radix_histogram!(counts :: VI, codes :: VU, v :: Val{N}) where {T <: Unsigned, N, VI <: AbstractVector{Int}, VU <: AbstractVector{T}}
    fill!(counts, 0)
    @inbounds for i in eachindex(codes)
        bucket = _radix_bucket(codes[i], v)
        counts[bucket] += 1
    end
    return nothing
end
