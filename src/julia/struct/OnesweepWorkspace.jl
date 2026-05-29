struct OnesweepWorkspace{KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}, OffsetV <: AbstractVector{UInt32}}
    # Global storages
    ## Workspace for data
    dst :: KeyV

    ## Workspace for sortperm indices.
    ## `perms[1]` and `perms[2]` are ping-pong buffers.
    ## They store 1-based source indices.
    perms :: NTuple{2, OffsetV}

    ## Counter; length-1 vector, corresponds to CUB d_ctrs
    tile_counter :: OffsetV

    ## Packed look-back table; flat vector, corresponds to CUB d_lookback
    ##
    ## Each entry uses the highest two bits as state:
    ##   entry == 0                 => EMPTY
    ##   01xxxxxx...xxxx            => PARTIAL local count
    ##   10xxxxxx...xxxx            => GLOBAL prefix count
    ##
    ## The lower 30 bits store the count.
    ## Length = 256 * ntiles for an 8-bit radix pass.
    lookback :: OffsetV

    ## Bucket offsets; correspond to CUB d_bins_in / d_bins_out.
    ##
    ## Both buffers store 1-based Julia output indices.
    ## bucket_offsets[1][bucket + 1] and bucket_offsets[2][bucket + 1]
    ## are the first Julia indices for `bucket`, depending on the pass.
    ##
    ## Odd passes read bucket_offsets[1] and write bucket_offsets[2].
    ## Even passes read bucket_offsets[2] and write bucket_offsets[1].
    ##
    ## Each buffer has length = 256 for an 8-bit radix pass.
    bucket_offsets :: NTuple{2, OffsetV}

    # Temporary storages
    ## Per-worker local histograms.
    ## Corresponds to CUB's per-agent `bins`.
    ## local_counts[(worker_id - 1) * 256 + bucket] stores the local count
    ## for `bucket - 1` in the current tile handled by this worker.
    ## Length = 256 * nworkers.
    local_counts :: OffsetV

    ## Per-worker local/exclusive digit offsets.
    ## Corresponds to CUB's `exclusive_digit_prefix`.
    ## local_offsets[(worker_id - 1) * 256 + bucket] stores the tile-local
    ## exclusive offset for `bucket - 1`.
    ## Length = 256 * nworkers.
    local_offsets :: OffsetV

    ## Per-worker global output offsets.
    ## Corresponds to CUB TempStorage_::global_offsets.
    ## global_offsets[(worker_id - 1) * 256 + bucket] stores the global
    ## output base for `bucket - 1` for the current tile.
    ## Length = 256 * nworkers.
    global_offsets :: OffsetV

    ## Per-worker local ranks.
    ## Corresponds to CUB's per-thread `ranks`, but stored as a flat
    ## per-worker scratch buffer in this CPU implementation.
    ## local_ranks[(worker_id - 1) * TileSize + local_i] stores the local
    ## rank of the `local_i`-th element in the current tile.
    ## Length = TileSize * nworkers.
    local_ranks :: OffsetV

    function OnesweepWorkspace(:: Type{KeyV}) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}}
        # Global workspaces
        dst = KeyV(undef, 0)

        tile_counter = similar(dst, UInt32, 0)
        lookback = similar(dst, UInt32, 0)

        bucket_offsets_1 = similar(dst, UInt32, 0)
        bucket_offsets_2 = similar(dst, UInt32, 0)

        perm_1 = similar(dst, UInt32, 0)
        perm_2 = similar(dst, UInt32, 0)

        # Temporary workspaces
        local_counts   = similar(dst, UInt32, 0)
        local_offsets  = similar(dst, UInt32, 0)
        global_offsets = similar(dst, UInt32, 0)
        local_ranks    = similar(dst, UInt32, 0)

        # Type stablizer
        OffsetV  = typeof(bucket_offsets_1)

        return new{KeyT, KeyV, OffsetV}(
            dst,
            (perm_1, perm_2),
            tile_counter,
            lookback,
            (bucket_offsets_1, bucket_offsets_2),
            local_counts,
            local_offsets,
            global_offsets,
            local_ranks
        )
    end
end

function initialize_workspace!(ws :: OnesweepWorkspace, nelems :: Int, ntiles :: Int, :: Val{NWorkers}, :: Val{TileSize}) where {NWorkers, TileSize}
    
    resize!(ws.dst, nelems)
    resize!(ws.tile_counter, 1)
    resize!(ws.lookback, 256 * ntiles)

    resize!(ws.bucket_offsets[1], 256)
    resize!(ws.bucket_offsets[2], 256)

    resize!(ws.local_counts,  256 * NWorkers)
    resize!(ws.local_offsets, 256 * NWorkers)
    resize!(ws.global_offsets, 256 * NWorkers)
    resize!(ws.local_ranks,   TileSize * NWorkers)

    fill!(ws.tile_counter, zero(UInt32))
    fill!(ws.lookback, zero(UInt32))
    fill!(ws.bucket_offsets[1], zero(UInt32))
    fill!(ws.bucket_offsets[2], zero(UInt32))

    fill!(ws.local_counts, zero(UInt32))
    fill!(ws.local_offsets, zero(UInt32))
    fill!(ws.global_offsets, zero(UInt32))
    fill!(ws.local_ranks, zero(UInt32))

    return nothing
end

function initialize_perm_indices!(perm :: Vector{UInt32}, nelems :: Int)
    @inbounds for i in 1:nelems
        perm[i] = UInt32(i)
    end
    return nothing
end

function initialize_perm_workspace!(ws :: OnesweepWorkspace{KeyT, KeyV}, nelems :: Int, ntiles :: Int, :: Val{NWorkers}, :: Val{TileSize}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, NWorkers, TileSize}
    # Initialize the common key-sorting workspace.
    initialize_workspace!(ws, nelems, ntiles, Val(NWorkers), Val(TileSize))

    # Resize the ping-pong permutation buffers.
    # perms[1] is initialized as the input permutation buffer.
    # perms[2] is used as the output permutation buffer for the first pass.
    resize!(ws.perms[1], nelems)
    resize!(ws.perms[2], nelems)
    
    # Fill the initial permutation with 1-based Julia indices.
    initialize_perm_indices!(ws.perms[1], nelems)

    return nothing
end

function reset_pass_workspace!(ws :: OnesweepWorkspace)
    fill!(ws.tile_counter, zero(UInt32))
    fill!(ws.lookback, zero(UInt32))
    return nothing
end