"""
    initialize_perm_indices!(perm::Vector{UInt32}, nelems::Int)

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

@inline function clear_base_workspace_kernel!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}) where {KeyT <: Unsigned, KeyV <: MtlDeviceVector{KeyT}, OffsetV <: MtlDeviceVector{UInt32}}
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    i = tid
    while i <= length(ws.tile_counter)
        @inbounds ws.tile_counter[i] = zero(UInt32)
        i += stride
    end

    i = tid
    while i <= length(ws.lookback)
        @inbounds ws.lookback[i] = zero(UInt32)
        i += stride
    end

    i = tid
    while i <= length(ws.bucket_offsets)
        @inbounds ws.bucket_offsets[i] = zero(UInt32)
        i += stride
    end

    return nothing
end

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

    return nothing
end

function UnsignedRadixSorts.initialize_base_workspace!(
        ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV},
        nelems :: Int,
        ntiles :: Int,
    ) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}}
    resize_base_workspace!(ws, nelems, ntiles)
    @metal threads=(256,) groups=(1,) clear_base_workspace_kernel!(ws)
    return nothing
end

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

function initialize_perm_workspace_for_sort!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems :: Int, ntiles :: Int, :: Val{NWorkers}, :: Val{ThreadsPerWorker}) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}, NWorkers, ThreadsPerWorker}
    resize_base_workspace!(ws, nelems, ntiles)
    resize!(ws.perms[1], nelems)
    resize!(ws.perms[2], nelems)
    @metal threads=(ThreadsPerWorker,) groups=(NWorkers,) initialize_perm_indices!(ws.perms[1], nelems)
    Metal.synchronize()

    return nothing
end

"""
    initialize_perm_workspace!(ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems::Int, ntiles::Int, ::Val{NWorkers}, ::Val{ThreadsPerWorker}, ::Val{TileSize}) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}, NWorkers, ThreadsPerWorker, TileSize}

Resize and clear the Onesweep workspace for permutation sorting.

# Parameters

- `ws`: Workspace whose key-sorting and permutation buffers are resized and initialized.
- `nelems`: Number of elements to sort.
- `ntiles`: Number of tiles processed by a radix pass.
- `::Val{NWorkers}`: Compile-time number of Metal threadgroups used for permutation initialization.
- `::Val{ThreadsPerWorker}`: Compile-time number of Metal threads used to initialize permutation indices.
- `::Val{TileSize}`: Compile-time tile size used by the later pass kernel's threadgroup-local rank storage.
"""
function UnsignedRadixSorts.initialize_perm_workspace!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems :: Int, ntiles :: Int, :: Val{NWorkers}, :: Val{ThreadsPerWorker}, :: Val{TileSize}) where {KeyT <: Unsigned, KeyV <: MtlVector{KeyT}, OffsetV <: MtlVector{UInt32}, NWorkers, ThreadsPerWorker, TileSize}
    # Initialize only the base workspace; GPU kernels keep local scratch in
    # threadgroup memory instead of ws.local_* buffers.
    initialize_base_workspace!(ws, nelems, ntiles)

    # Resize the ping-pong permutation buffers.
    # perms[1] is initialized as the input permutation buffer.
    # perms[2] is used as the output permutation buffer for the first pass.
    resize!(ws.perms[1], nelems)
    resize!(ws.perms[2], nelems)
    
    # Fill the initial permutation with 1-based Julia indices.
    @metal threads=(ThreadsPerWorker,) groups=(NWorkers,) initialize_perm_indices!(ws.perms[1], nelems)
    Metal.synchronize()

    return nothing
end
