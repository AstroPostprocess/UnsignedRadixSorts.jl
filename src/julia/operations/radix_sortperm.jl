"""
    radix_sortperm!(codes :: V) where {T <: Unsigned, V <: AbstractVector{T}}
    radix_sortperm!(codes :: V, ws :: RadixSortPermWorkspace{T}) where {T <: Unsigned, V <: AbstractVector{T}}

Sort `codes` in-place in ascending order using an 8-bit LSD radix sort, and
return the corresponding permutation vector.

The high-level method `radix_sortperm!(codes)` allocates a fresh workspace for
the call and then invokes the low-level method `radix_sortperm!(codes, ws)`.

The low-level workspace method performs the same stable radix-sort passes but
reuses the temporary key and permutation buffers stored in `ws`, so the sorting
core does not need to allocate when the workspace size matches `codes`.

After sorting, `order[i]` gives the original position of the element now stored
at `codes[i]`. Equivalently, if `codes0` denotes the original input, then after
sorting `codes[i] == codes0[order[i]]`.

This method family is implemented for `UInt8`, `UInt16`, `UInt32`, `UInt64`,
and `UInt128`.

# Parameters

- `codes :: V`: Input vector of unsigned integer keys to be sorted in-place.
- `ws :: RadixSortPermWorkspace{T}`: Preallocated workspace containing temporary
  key buffers, permutation buffers, histogram buffer, and offset buffer.

# Returns

- `order :: Vector{Int}`: Permutation vector describing the sorted arrangement.
  For the workspace method this is `ws.order`, which is overwritten if the same
  workspace is reused by a later call.
"""
function radix_sortperm! end

function radix_sortperm!(codes :: V) where {V <: AbstractVector{UInt8}}
    return radix_sortperm!(codes, RadixSortPermWorkspace(UInt8, length(codes)))
end

function radix_sortperm!(codes :: V) where {V <: AbstractVector{UInt16}}
    return radix_sortperm!(codes, RadixSortPermWorkspace(UInt16, length(codes)))
end

function radix_sortperm!(codes :: V) where {V <: AbstractVector{UInt32}}
    return radix_sortperm!(codes, RadixSortPermWorkspace(UInt32, length(codes)))
end

function radix_sortperm!(codes :: V) where {V <: AbstractVector{UInt64}}
    return radix_sortperm!(codes, RadixSortPermWorkspace(UInt64, length(codes)))
end

function radix_sortperm!(codes :: V) where {V <: AbstractVector{UInt128}}
    return radix_sortperm!(codes, RadixSortPermWorkspace(UInt128, length(codes)))
end

function radix_sortperm!(codes :: V, ws :: RadixSortPermWorkspace{UInt8}) where {V <: AbstractVector{UInt8}}
    _check_workspace_size(codes, ws)
    _initialize_order!(ws.order)

    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes, ws.order, Val(1))
    copyto!(codes, ws.tmp_codes)
    copyto!(ws.order, ws.tmp_order)
    return ws.order
end

function radix_sortperm!(codes :: V, ws :: RadixSortPermWorkspace{UInt16}) where {V <: AbstractVector{UInt16}}
    _check_workspace_size(codes, ws)
    _initialize_order!(ws.order)

    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(1))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(2))
    return ws.order
end

function radix_sortperm!(codes :: V, ws :: RadixSortPermWorkspace{UInt32}) where {V <: AbstractVector{UInt32}}
    _check_workspace_size(codes, ws)
    _initialize_order!(ws.order)

    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(1))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(2))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(3))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(4))
    return ws.order
end

function radix_sortperm!(codes :: V, ws :: RadixSortPermWorkspace{UInt64}) where {V <: AbstractVector{UInt64}}
    _check_workspace_size(codes, ws)
    _initialize_order!(ws.order)

    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(1))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(2))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(3))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(4))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(5))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(6))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(7))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(8))
    return ws.order
end

function radix_sortperm!(codes :: V, ws :: RadixSortPermWorkspace{UInt128}) where {V <: AbstractVector{UInt128}}
    _check_workspace_size(codes, ws)
    _initialize_order!(ws.order)

    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(1))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(2))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(3))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(4))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(5))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(6))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(7))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(8))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(9))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(10))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(11))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(12))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(13))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(14))
    radix_pass!(ws.tmp_codes, ws.tmp_order, ws.counts, ws.offsets, codes,        ws.order,     Val(15))
    radix_pass!(codes,        ws.order,     ws.counts, ws.offsets, ws.tmp_codes, ws.tmp_order, Val(16))
    return ws.order
end
