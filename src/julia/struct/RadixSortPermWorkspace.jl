"""
    RadixSortPermWorkspace{T <: Unsigned, TI <: Integer, VT <: AbstractVector{T}, VO <: AbstractVector{Int}, MI <: AbstractMatrix{TI}}

Workspace for allocation-free calls to [`radix_sortperm!`](@ref).

The workspace owns the temporary buffers required by the permutation-tracking
radix-sort core: temporary key storage `tmp_codes`, permutation buffers
`order` and `tmp_order`, plus the shared histogram and offset buffers.
The returned permutation from `radix_sortperm!(codes, ws)` is `ws.order`, so
it is overwritten by the next call that reuses the same workspace.

# Fields

- `tmp_codes :: VT`: Temporary key buffer used by alternating radix passes.
- `order :: VO`: Permutation buffer describing the sorted arrangement.
- `tmp_order :: VO`: Temporary permutation buffer paired with `tmp_codes`.
- `counts :: MI`: Histogram buffer for the 256 radix buckets in each chunk.
- `offsets :: MI`: Buffer for 1-based bucket start positions and scatter cursors in each chunk.
"""
struct RadixSortPermWorkspace{T <: Unsigned, TI <: Integer, VT <: AbstractVector{T}, VO <: AbstractVector{Int}, MI <: AbstractMatrix{TI}}
    nchunks   :: Int
    tmp_codes :: VT
    order     :: VO
    tmp_order :: VO
    counts    :: MI
    offsets   :: MI
end

"""
    RadixSortPermWorkspace(::Type{T}, n :: Integer) where {T <: Unsigned}

Construct a workspace for allocation-free calls to [`radix_sortperm!`](@ref)
on vectors of `n` unsigned elements of type `T`.

# Parameters

- `T`: Unsigned integer element type to be sorted.
- `n :: Integer`: Required vector length for the workspace.
"""
function RadixSortPermWorkspace(::Type{T}, n :: Integer) where {T <: Unsigned}
    n >= 0 || throw(ArgumentError("workspace length must be non-negative, got $n"))
    nchunks = Threads.nthreads()
    return RadixSortPermWorkspace(
        nchunks,
        Vector{T}(undef, n),
        Vector{Int}(undef, n),
        Vector{Int}(undef, n),
        Matrix{Int}(undef, 256, nchunks),
        Matrix{Int}(undef, 256, nchunks),
    )
end

@inline function _check_workspace_size(codes :: V, ws :: RadixSortPermWorkspace{T}) where {T <: Unsigned, V <: AbstractVector{T}}
    n = length(codes)
    length(ws.tmp_codes) == n ||
        throw(DimensionMismatch("workspace tmp_codes length $(length(ws.tmp_codes)) does not match codes length $n"))
    length(ws.order) == n ||
        throw(DimensionMismatch("workspace order length $(length(ws.order)) does not match codes length $n"))
    length(ws.tmp_order) == n ||
        throw(DimensionMismatch("workspace tmp_order length $(length(ws.tmp_order)) does not match codes length $n"))
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

@inline function _initialize_order!(order :: V) where {V <: AbstractVector{Int}}
    @inbounds for i in eachindex(order)
        order[i] = i
    end
    return nothing
end
