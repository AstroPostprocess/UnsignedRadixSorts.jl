"""
    RadixSortWorkspace{T <: Unsigned, TI <: Integer, VT <: AbstractVector{T}, MI <: AbstractMatrix{TI}}

Workspace for allocation-free calls to [`radix_sort!`](@ref).

The workspace owns the temporary buffers required by the radix-sort core:
a temporary key buffer `tmp`, a histogram buffer `counts`, and a bucket-offset
buffer `offsets`. Reusing the same workspace across repeated sorts avoids
reallocating these arrays on every call.

# Fields

- `tmp :: VT`: Temporary key buffer used to hold the output of alternating radix passes.
- `counts :: MI`: Histogram buffer for the 256 radix buckets in each chunk.
- `offsets :: MI`: Buffer for 1-based bucket start positions and scatter cursors in each chunk.
"""
struct RadixSortWorkspace{T <: Unsigned, TI <: Integer, VT <: AbstractVector{T}, MI <: AbstractMatrix{TI}}
    nchunks  :: Int
    tmp      :: VT
    counts   :: MI
    offsets  :: MI
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
    nchunks = Threads.nthreads()
    return RadixSortWorkspace(
        nchunks,
        Vector{T}(undef, n),
        Matrix{Int}(undef, 256, nchunks),
        Matrix{Int}(undef, 256, nchunks),
    )
end

@inline function _check_workspace_size(codes :: V, ws :: RadixSortWorkspace{T}) where {T <: Unsigned, V <: AbstractVector{T}}
    length(ws.tmp) == length(codes) ||
        throw(DimensionMismatch("workspace length $(length(ws.tmp)) does not match codes length $(length(codes))"))
    size(ws.counts, 1) == 256 ||
        throw(DimensionMismatch("workspace counts must have 256 rows"))
    size(ws.offsets, 1) == 256 ||
        throw(DimensionMismatch("workspace offsets must have 256 rows"))
    ws.nchunks >= 1 ||
        throw(DimensionMismatch("workspace nchunks must be at least 1"))
    size(ws.counts, 2) >= 1 ||
        throw(DimensionMismatch("workspace counts must have at least one chunk column"))
    size(ws.offsets, 2) >= 1 ||
        throw(DimensionMismatch("workspace offsets must have at least one chunk column"))
    size(ws.counts, 2) >= ws.nchunks ||
        throw(DimensionMismatch("workspace counts columns $(size(ws.counts, 2)) do not cover nchunks $(ws.nchunks)"))
    size(ws.offsets, 2) >= ws.nchunks ||
        throw(DimensionMismatch("workspace offsets columns $(size(ws.offsets, 2)) do not cover nchunks $(ws.nchunks)"))
    return nothing
end
