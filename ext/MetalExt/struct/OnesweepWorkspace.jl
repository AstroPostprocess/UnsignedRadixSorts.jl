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
