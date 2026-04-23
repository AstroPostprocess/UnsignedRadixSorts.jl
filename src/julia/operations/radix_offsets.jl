"""
    radix_offsets!(offsets :: VO, counts :: VI) where {VO <: AbstractVector{Int}, VI <: AbstractVector{Int}}

Compute the 1-based exclusive prefix offsets from a radix histogram and write the result in-place to `offsets`.

For each bucket index `i`, `offsets[i]` stores the first output position assigned to that bucket during the scatter step of a radix-sort pass. The function assumes that `counts[i]` contains the number of elements assigned to bucket `i`, and transforms these bucket counts into starting write positions using Julia's 1-based indexing convention.

For example, if `counts == [2, 3, 1, 2]`, then the resulting offsets are `[1, 3, 6, 7]`.

# Parameters

- `offsets :: VO`: Preallocated output buffer that will be overwritten in-place with the bucket start positions. It is expected to have the same length as `counts`.
- `counts :: VI`: Radix histogram counts for a single pass, where `counts[i]` is the number of elements assigned to bucket `i`.
"""
@inline function radix_offsets!(offsets :: VO, counts :: VI) where {VO <: AbstractVector{Int}, VI <: AbstractVector{Int}}
    pos = 1
    @inbounds for i in eachindex(counts)
        offsets[i] = pos
        pos += counts[i]
    end
    return nothing
end
