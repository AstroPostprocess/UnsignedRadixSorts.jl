"""
    RadixSortWorkspace{T <: Unsigned}

Workspace for allocation-free calls to [`radix_sort!`](@ref).

The workspace owns the temporary buffers required by the radix-sort core:
a temporary key buffer `tmp`, a histogram buffer `counts`, and a bucket-offset
buffer `offsets`. Reusing the same workspace across repeated sorts avoids
reallocating these arrays on every call.

# Fields

- `tmp :: Vector{T}`: Temporary key buffer used to hold the output of alternating radix passes.
- `counts :: Vector{Int}`: Histogram buffer for the 256 radix buckets.
- `offsets :: Vector{Int}`: Buffer for 1-based bucket start positions and scatter cursors.
"""
struct RadixSortWorkspace{T <: Unsigned}
    tmp     :: Vector{T}
    counts  :: Vector{Int}
    offsets :: Vector{Int}
end

"""
    RadixSortWorkspace(::Type{T}, n :: Integer) where {T <: Unsigned}

Construct a workspace for allocation-free calls to [`radix_sort!`](@ref) on
vectors of `n` unsigned elements of type `T`.

# Parameters

- `T`: Unsigned integer element type to be sorted.
- `n :: Integer`: Required vector length for the workspace.
"""
function RadixSortWorkspace(::Type{T}, n :: Integer) where {T <: Unsigned}
    n >= 0 || throw(ArgumentError("workspace length must be non-negative, got $n"))
    return RadixSortWorkspace(
        Vector{T}(undef, n),
        Vector{Int}(undef, 256),
        Vector{Int}(undef, 256),
    )
end

@inline function _check_workspace_size(codes :: V, ws :: RadixSortWorkspace{T}) where {T <: Unsigned, V <: AbstractVector{T}}
    length(ws.tmp) == length(codes) ||
        throw(DimensionMismatch("workspace length $(length(ws.tmp)) does not match codes length $(length(codes))"))
    length(ws.counts) == 256 ||
        throw(DimensionMismatch("workspace counts length must be 256"))
    length(ws.offsets) == 256 ||
        throw(DimensionMismatch("workspace offsets length must be 256"))
    return nothing
end
