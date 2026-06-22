"""
    initialize_perm_indices!(perm::CuVector{UInt32}, nelems::Int)

Fill a permutation buffer with 1-based Julia source indices.

# Parameters

- `perm`: Permutation buffer to initialize.
- `nelems`: Number of indices to write.
"""
@inline function UnsignedRadixSorts.initialize_perm_indices!(perm :: CuDeviceVector{UInt32}, nelems :: Int)
    tid = Int((CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x)
    stride = Int(CUDA.gridDim().x * CUDA.blockDim().x)

    i = tid
    while i <= nelems
        perm[i] = UInt32(i)

        i += stride
    end
    return nothing
end

"""
    initialize_perm_workspace!(ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems::Int, ntiles::Int, ::Val{NBlocks}, ::Val{ThreadsPerBlock}, ::Val{TileSize}) where {KeyT <: Unsigned, KeyV <: CuVector{KeyT}, OffsetV <: CuVector{UInt32}, NBlocks, ThreadsPerBlock, TileSize}

Resize and clear the Onesweep workspace for permutation sorting.

# Parameters

- `ws`: Workspace whose key-sorting and permutation buffers are resized and initialized.
- `nelems`: Number of elements to sort.
- `ntiles`: Number of tiles processed by a radix pass.
- `::Val{NBlocks}`: Compile-time number of CUDA blocks used for scratch storage.
- `::Val{ThreadsPerBlock}`: Compile-time number of CUDA threads per block used to initialize permutation indices.
- `::Val{TileSize}`: Compile-time tile size used for local rank storage.
"""
function UnsignedRadixSorts.initialize_perm_workspace!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems :: Int, ntiles :: Int, :: Val{NBlocks}, :: Val{ThreadsPerBlock}, :: Val{TileSize}) where {KeyT <: Unsigned, KeyV <: CuVector{KeyT}, OffsetV <: CuVector{UInt32}, NBlocks, ThreadsPerBlock, TileSize}
    # Initialize the common key-sorting workspace.
    initialize_workspace!(ws, nelems, ntiles, Val(NBlocks), Val(TileSize))

    # Resize the ping-pong permutation buffers.
    # perms[1] is initialized as the input permutation buffer.
    # perms[2] is used as the output permutation buffer for the first pass.
    resize!(ws.perms[1], nelems)
    resize!(ws.perms[2], nelems)
    
    # Fill the initial permutation with 1-based Julia indices.
    CUDA.@cuda threads=ThreadsPerBlock blocks=NBlocks initialize_perm_indices!(ws.perms[1], nelems)
    CUDA.synchronize()

    return nothing
end
