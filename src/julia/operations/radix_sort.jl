"""
    radix_sort!(codes :: V) where {T <: Unsigned, V <: AbstractVector{T}}
    radix_sort!(codes :: V, ws :: RadixSortWorkspace{T}) where {T <: Unsigned, V <: AbstractVector{T}}

Sort `codes` in-place in ascending order using an 8-bit LSD radix sort.

The high-level method `radix_sort!(codes)` allocates a fresh workspace for the
call and then invokes the low-level method `radix_sort!(codes, ws)`.

The low-level workspace method performs the same stable radix-sort passes but
reuses the temporary buffers stored in `ws`, so the sorting core does not need
to allocate when the workspace size matches `codes`.

This method family is implemented for `UInt8`, `UInt16`, `UInt32`, `UInt64`,
and `UInt128`.

# Parameters

- `codes :: V`: Input vector of unsigned integer keys to be sorted in-place.
- `ws :: RadixSortWorkspace{T}`: Preallocated workspace containing the temporary
  key buffer, histogram buffer, and offset buffer used by the radix-sort core.
"""
function radix_sort! end

function radix_sort!(codes :: V) where {V <: AbstractVector{UInt8}}
    return radix_sort!(codes, RadixSortWorkspace(UInt8, length(codes)))
end

function radix_sort!(codes :: V) where {V <: AbstractVector{UInt16}}
    return radix_sort!(codes, RadixSortWorkspace(UInt16, length(codes)))
end

function radix_sort!(codes :: V) where {V <: AbstractVector{UInt32}}
    return radix_sort!(codes, RadixSortWorkspace(UInt32, length(codes)))
end

function radix_sort!(codes :: V) where {V <: AbstractVector{UInt64}}
    return radix_sort!(codes, RadixSortWorkspace(UInt64, length(codes)))
end

function radix_sort!(codes :: V) where {V <: AbstractVector{UInt128}}
    return radix_sort!(codes, RadixSortWorkspace(UInt128, length(codes)))
end

function radix_sort!(codes :: V, ws :: RadixSortWorkspace{UInt8}) where {V <: AbstractVector{UInt8}}
    _check_workspace_size(codes, ws)

    radix_pass!(ws.tmp, ws.counts, ws.offsets, codes, Val(1))
    copyto!(codes, ws.tmp)
    return nothing
end

function radix_sort!(codes :: V, ws :: RadixSortWorkspace{UInt16}) where {V <: AbstractVector{UInt16}}
    _check_workspace_size(codes, ws)

    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(1))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(2))
    return nothing
end

function radix_sort!(codes :: V, ws :: RadixSortWorkspace{UInt32}) where {V <: AbstractVector{UInt32}}
    _check_workspace_size(codes, ws)

    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(1))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(2))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(3))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(4))
    return nothing
end

function radix_sort!(codes :: V, ws :: RadixSortWorkspace{UInt64}) where {V <: AbstractVector{UInt64}}
    _check_workspace_size(codes, ws)

    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(1))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(2))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(3))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(4))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(5))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(6))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(7))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(8))
    return nothing
end

function radix_sort!(codes :: V, ws :: RadixSortWorkspace{UInt128}) where {V <: AbstractVector{UInt128}}
    _check_workspace_size(codes, ws)

    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(1))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(2))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(3))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(4))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(5))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(6))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(7))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(8))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(9))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(10))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(11))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(12))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(13))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(14))
    radix_pass!(ws.tmp,   ws.counts, ws.offsets, codes,  Val(15))
    radix_pass!(codes,    ws.counts, ws.offsets, ws.tmp, Val(16))
    return nothing
end
