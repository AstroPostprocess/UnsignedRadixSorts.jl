"""
    radix_pass!(out :: VO, counts :: VI, offsets :: VI, codes :: VU, v :: Val{N}) where {T <: Unsigned, N, VO <: AbstractVector{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{T}}

Perform one stable 8-bit LSD radix-sort pass on `codes` for pass `N`, writing the reordered result to `out`.

The pass consists of three stages: histogram construction, conversion of histogram counts to 1-based exclusive bucket offsets, and stable scatter into `out`. The `counts` buffer is overwritten during histogram construction. The `offsets` buffer is first written with bucket start positions and then mutated during scattering as a set of per-bucket write cursors.

# Parameters

- `out :: VO`: Output buffer receiving the reordered values. It is expected to have the same length as `codes`.
- `counts :: VI`: Preallocated histogram buffer for the 256 radix buckets. It is overwritten in-place.
- `offsets :: VI`: Preallocated buffer for bucket start positions and scatter cursors. It is overwritten in-place.
- `codes :: VU`: Input vector of unsigned radix keys for this pass.
- `v :: Val{N}`: Compile-time pass selector indicating which 8-bit digit to use. `Val(1)` selects the least significant byte, `Val(2)` the next byte, and so on.
"""
@inline function radix_pass!(out :: VO, counts :: VI, offsets :: VI, codes :: VU, v :: Val{N}) where {T <: Unsigned, N, VO <: AbstractVector{T}, VI <: AbstractVector{Int}, VU <: AbstractVector{T}}
    radix_histogram!(counts, codes, v)
    radix_offsets!(offsets, counts)
    radix_scatter!(out, offsets, codes, v)
    return nothing
end



"""
    radix_pass!(out_codes :: VO, out_order :: VOO, counts :: VI, offsets :: VI, codes :: VC, order :: VOI, v :: Val{N}) where {N, T <: Unsigned, VO <: AbstractVector{T}, VC <: AbstractVector{T}, VOO <: AbstractVector{Int}, VOI <: AbstractVector{Int}, VI <: AbstractVector{Int}}

Perform one stable 8-bit LSD radix-sort pass on `codes`, writing the reordered keys to `out_codes` and the corresponding permutation indices to `out_order`.

The pass consists of three stages: histogram construction, conversion of histogram counts to 1-based exclusive bucket offsets, and stable scatter. The input vectors `codes` and `order` are read as the source arrangement for the current pass. The output vectors `out_codes` and `out_order` receive the reordered keys and permutation indices for pass `N`. The buffers `counts` and `offsets` are overwritten in-place during the pass.

# Parameters

- `out_codes :: VO`: Output buffer receiving the reordered unsigned keys for this pass.
- `out_order :: VOO`: Output buffer receiving the reordered permutation indices corresponding to `out_codes`.
- `counts :: VI`: Preallocated histogram buffer for the 256 radix buckets. It is overwritten in-place.
- `offsets :: VI`: Preallocated buffer used first for bucket start positions and then as per-bucket scatter cursors. It is overwritten in-place.
- `codes :: VC`: Input buffer containing the unsigned keys in the current source order.
- `order :: VOI`: Input permutation buffer associated with `codes`. Each entry tracks the original position of the corresponding key.
- `v :: Val{N}`: Compile-time pass selector indicating which 8-bit digit to use. `Val(1)` selects the least significant byte, `Val(2)` the next byte, and so on.
"""
@inline function radix_pass!(out_codes :: VO, out_order :: VOO, counts :: VI, offsets :: VI, codes :: VC, order :: VOI, v :: Val{N}) where {N, T <: Unsigned, VO <: AbstractVector{T}, VC <: AbstractVector{T}, VOO <: AbstractVector{Int}, VOI <: AbstractVector{Int}, VI <: AbstractVector{Int}}
    radix_histogram!(counts, codes, v)
    radix_offsets!(offsets, counts)
    radix_scatter!(out_codes, out_order, offsets, codes, order, v)
    return nothing
end
