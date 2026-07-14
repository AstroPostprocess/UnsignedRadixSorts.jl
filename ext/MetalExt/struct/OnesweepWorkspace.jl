"""
    initialize_perm_indices!(perm::MtlVector{UInt32}, nelems::Int)

Fill a permutation buffer with 1-based Julia source indices.

# Parameters

- `perm`: Permutation buffer to initialize.
- `nelems`: Number of indices to write.
"""
@inline function UnsignedRadixSorts.initialize_perm_indices!(perm :: MtlDeviceVector{UInt32}, nelems :: Int)
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    i = tid
    while i <= nelems
        perm[i] = UInt32(i)

        i += stride
    end
    return nothing
end

"""
    resize_base_workspace!(ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems::Int, ntiles::Int) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}}

Resize the Metal OneSweep base workspace for one sorting problem.

CUB/CUDA parallel: this is the persistent global workspace used by the
dispatch layer around AgentRadixSortOnesweep.  Per-tile scratch remains in
threadgroup memory inside the pass kernel.

# Parameters

- `ws`: Workspace whose key destination and lookback buffers are resized.
- `nelems`: Number of elements to sort.
- `ntiles`: Number of tiles processed by one radix pass.
"""
function resize_base_workspace!(
        ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV},
        nelems :: Int,
        ntiles :: Int,
    ) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}}
    nelems <= Int(typemax(UInt32) >> 2) ||
        throw(ArgumentError("OneSweep lookback supports at most $(typemax(UInt32) >> 2) elements"))

    npass = UnsignedRadixSorts._npasses(KeyT)

    resize!(ws.dst, nelems)
    resize!(ws.tile_counter, 1)
    resize!(ws.lookback, 256 * ntiles)
    resize!(ws.bucket_offsets, 256 * npass)
    # Metal's forward-progress-safe pass persists each element's rank within
    # its radix bucket across the rank, prefix, and scatter dispatches.
    resize!(ws.local_ranks, nelems)

    return nothing
end

"""
    clear_pass_workspace_kernel!(tile_counter, lookback)

Clear the per-pass global workspace used by OneSweep lookback.

This kernel resets the dynamic tile counter and the packed lookback table before
one radix digit pass starts.  Bucket offsets are not cleared here because the
prepass histogram/scan writes them once for all passes.
"""
@inline function clear_pass_workspace_kernel!(tile_counter :: OffsetV, lookback :: OffsetV) where {OffsetV <: MtlDeviceVector{UInt32}}
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    i = tid
    while i <= length(tile_counter)
        @inbounds tile_counter[i] = zero(UInt32)
        i += stride
    end

    i = tid
    while i <= length(lookback)
        @inbounds lookback[i] = zero(UInt32)
        i += stride
    end

    return nothing
end

function UnsignedRadixSorts.reset_pass_workspace!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}}
    @metal threads=(256,) groups=(1,) clear_pass_workspace_kernel!(ws.tile_counter, ws.lookback)
    return nothing
end

"""
    initialize_perm_workspace_for_sort!(ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems::Int, ntiles::Int, ::Val{NThreadgroups}, ::Val{ThreadsPerGroup}) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}, NThreadgroups, ThreadsPerGroup}

Resize and initialize the Metal OneSweep workspace for permutation sorting.

# Parameters

- `ws`: Workspace whose key-sorting and permutation buffers are resized and initialized.
- `nelems`: Number of elements to sort.
- `ntiles`: Number of tiles processed by a radix pass.
- `::Val{NThreadgroups}`: Compile-time number of Metal threadgroups used for permutation initialization.
- `::Val{ThreadsPerGroup}`: Compile-time number of Metal threads per threadgroup used to initialize permutation indices.
"""
function initialize_perm_workspace_for_sort!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems :: Int, ntiles :: Int, :: Val{NThreadgroups}, :: Val{ThreadsPerGroup}) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}, NThreadgroups, ThreadsPerGroup}
    # Initialize only the base workspace; pass kernels keep local scratch in
    # threadgroup memory instead of ws.local_* buffers.
    resize_base_workspace!(ws, nelems, ntiles)

    # Resize the ping-pong permutation buffers.
    # perms[1] is initialized as the input permutation buffer.
    # perms[2] is used as the output permutation buffer for the first pass.
    resize!(ws.perms[1], nelems)
    resize!(ws.perms[2], nelems)

    # Fill the initial permutation with 1-based Julia indices.
    @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) initialize_perm_indices!(ws.perms[1], nelems)
    Metal.synchronize()

    return nothing
end
