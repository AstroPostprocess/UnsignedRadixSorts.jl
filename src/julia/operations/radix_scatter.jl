"""
    radix_scatter!(out :: VO, offsets :: VI, codes :: VU, v :: Val{N}) where {N, T <: Unsigned, VI <: AbstractVector{Int}, VU <: AbstractVector{T}, VO <: AbstractVector{T}}

Stably scatter `codes` into `out` according to the `N`-th 8-bit radix digit.

For each element in `codes`, the function recomputes its radix bucket for pass `N`, writes the element to the next available output position for that bucket, and then advances the corresponding bucket offset. The `offsets` vector is updated in-place during this process and therefore acts as a set of per-bucket write cursors rather than immutable start positions.

# Parameters

- `out :: VO`: Output buffer receiving the reordered values.
- `offsets :: VI`: Per-bucket write positions. It is mutated in-place during scattering and is expected to contain the 1-based bucket start positions before the call.
- `codes :: VU`: Input vector of unsigned radix keys to scatter.
- `v :: Val{N}`: Compile-time pass selector indicating which 8-bit digit to use. `Val(1)` selects the least significant byte, `Val(2)` the next byte, and so on.
"""
@inline function radix_scatter!(out :: VO, offsets :: VI, codes :: VU, v :: Val{N}) where {N, T <: Unsigned, VO <: AbstractVector{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{T}}
    @inbounds for i in eachindex(codes)
        x = codes[i]
        bucket = _radix_bucket(x, v)
        j = offsets[bucket]
        out[j] = x
        offsets[bucket] = j + 1
    end
    return nothing
end


"""
    radix_scatter!(out_codes :: VC, out_order :: VO, offsets :: VI, codes :: VC, order :: VO, v :: Val{N}) where {N, T <: Unsigned, VC <: AbstractVector{T}, VO <: AbstractVector{Int}, VI <: AbstractVector{Int}}

Stably scatter `codes` and their associated permutation indices from the current pass input buffers into the corresponding output buffers for pass `N`.

For each element in `codes`, the function recomputes its radix bucket for the `N`-th 8-bit digit, writes the key to the next available output position for that bucket in `out_codes`, writes the associated source index from `order` to the same position in `out_order`, and then advances the corresponding bucket cursor in `offsets`. The `offsets` buffer is therefore mutated in-place during scattering and acts as a set of per-bucket write cursors rather than immutable bucket start positions.

# Parameters
- `out_codes :: VC`: Output buffer receiving the reordered unsigned keys for the current radix pass.
- `out_order :: VO`: Output buffer receiving the reordered permutation indices corresponding to `out_codes`.
- `offsets :: VI`: Per-bucket write positions for the current pass. It is mutated in-place during scattering and is expected to contain the 1-based bucket start positions before the call.
- `codes :: VC`: Input buffer containing the unsigned keys in the current source order.
- `order :: VO`: Input permutation buffer associated with `codes`. Each entry tracks the original position of the corresponding key.
- `v :: Val{N}`: Compile-time pass selector indicating which 8-bit digit to use. `Val(1)` selects the least significant byte, `Val(2)` the next byte, and so on.
"""
@inline function radix_scatter!(out_codes :: VC, out_order :: VO, offsets :: VI, codes :: VC, order :: VO, v :: Val{N}) where {N, T <: Unsigned, VC <: AbstractVector{T}, VO <: AbstractVector{Int}, VI <: AbstractVector{Int}}
    @inbounds for i in eachindex(codes)
        bucket = _radix_bucket(codes[i], v)
        j = offsets[bucket]
        out_codes[j] = codes[i]
        out_order[j] = order[i]
        offsets[bucket] = j + 1
    end
    return nothing
end
