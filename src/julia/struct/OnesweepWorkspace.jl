"""
    OnesweepWorkspace{KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}, OffsetV <: AbstractVector{UInt32}}

Workspace storage for Onesweep radix sort passes over unsigned integer keys.

# Type Parameters

- `KeyT`: Unsigned integer key type sorted by the workspace.
- `KeyV`: Abstract vector type used for key storage.
- `OffsetV`: Abstract vector type used for UInt32 counters, offsets, ranks, and permutation buffers.

# Fields

- `dst::KeyV`                   : Destination buffer for ping-pong key sorting.
- `perms::NTuple{2, OffsetV}`   : Ping-pong buffers for 1-based source permutation indices.
- `tile_counter::OffsetV`       : Length-one counter used to assign tiles during a pass.
- `lookback::OffsetV`           : Packed look-back table storing per-bucket tile prefix state and counts.
- `bucket_offsets::OffsetV`     : Global output start offsets for each bucket in each radix pass.
- `prepass_counts::OffsetV`     : Per-worker all-pass histograms used to build bucket offsets before Onesweep passes.
- `local_counts::OffsetV`       : Per-worker local bucket counts for the current tile.
- `local_offsets::OffsetV`      : Per-worker tile-local exclusive digit offsets.
- `global_offsets::OffsetV`     : Per-worker global output offsets for the current tile.
- `rank_cursors::OffsetV`       : Per-worker cursors used while assigning tile-local ranks.
- `local_ranks::OffsetV`        : Per-worker tile-local ranks for elements in the current tile.

"""
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

    ## Bucket offsets; corresponds to CUB d_bins.
    ## Stores 1-based Julia output start indices for every radix pass.
    ## bucket_offsets[(pass - 1) * 256 + bucket] is the first Julia output index
    ## for `bucket - 1` in `pass`, where `bucket ∈ 1:256`.
    ## Length = 256 * _npasses(KeyT) for an 8-bit radix pass.
    bucket_offsets :: OffsetV

    ## Per-worker all-pass histograms used before Onesweep passes.
    ## prepass_counts[(worker_id - 1) * _npasses(KeyT) * 256 +
    ##                (pass - 1) * 256 +
    ##                bucket]
    ## stores the local count for `bucket - 1` in `pass`, where `bucket ∈ 1:256`.
    ## Length = 256 * _npasses(KeyT) * nworkers.
    prepass_counts :: OffsetV

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

    ## Per-worker rank cursors.
    ## rank_cursors[(worker_id - 1) * 256 + bucket] is a temporary cursor
    ## used while assigning tile-local ranks for `bucket - 1`.
    ## Length = 256 * nworkers.
    rank_cursors :: OffsetV

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

        bucket_offsets = similar(dst, UInt32, 0)

        prepass_counts = similar(dst, UInt32, 0)

        perm_1 = similar(dst, UInt32, 0)
        perm_2 = similar(dst, UInt32, 0)

        # Temporary workspaces
        local_counts   = similar(dst, UInt32, 0)
        local_offsets  = similar(dst, UInt32, 0)
        global_offsets = similar(dst, UInt32, 0)
        rank_cursors   = similar(dst, UInt32, 0)
        local_ranks    = similar(dst, UInt32, 0)

        # Type stablizer
        OffsetV  = typeof(bucket_offsets)

        return new{KeyT, KeyV, OffsetV}(
            dst,
            (perm_1, perm_2),
            tile_counter,
            lookback,
            bucket_offsets,
            prepass_counts,
            local_counts,
            local_offsets,
            global_offsets,
            rank_cursors,
            local_ranks
        )
    end

    function OnesweepWorkspace(
            dst :: KeyV,
            perms :: NTuple{2, OffsetV},
            tile_counter :: OffsetV,
            lookback :: OffsetV,
            bucket_offsets :: OffsetV,
            prepass_counts :: OffsetV,
            local_counts :: OffsetV,
            local_offsets :: OffsetV,
            global_offsets :: OffsetV,
            rank_cursors :: OffsetV,
            local_ranks :: OffsetV,
        ) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}, OffsetV <: AbstractVector{UInt32}}

        return new{KeyT, KeyV, OffsetV}(
            dst,
            perms,
            tile_counter,
            lookback,
            bucket_offsets,
            prepass_counts,
            local_counts,
            local_offsets,
            global_offsets,
            rank_cursors,
            local_ranks
        )
    end
end

function Adapt.adapt_structure(to, x :: OnesweepWorkspace)
    dst = Adapt.adapt(to, x.dst)
    perms = (Adapt.adapt(to, x.perms[1]), Adapt.adapt(to, x.perms[2]))
    tile_counter = Adapt.adapt(to, x.tile_counter)
    lookback = Adapt.adapt(to, x.lookback)
    bucket_offsets = Adapt.adapt(to, x.bucket_offsets)
    prepass_counts = Adapt.adapt(to, x.prepass_counts)
    local_counts = Adapt.adapt(to, x.local_counts)
    local_offsets = Adapt.adapt(to, x.local_offsets)
    global_offsets = Adapt.adapt(to, x.global_offsets)
    rank_cursors = Adapt.adapt(to, x.rank_cursors)
    local_ranks = Adapt.adapt(to, x.local_ranks)

    return OnesweepWorkspace(
        dst,
        perms,
        tile_counter,
        lookback,
        bucket_offsets,
        prepass_counts,
        local_counts,
        local_offsets,
        global_offsets,
        rank_cursors,
        local_ranks,
    )
end

"""
    OnesweepWorkspace(::Type{KeyV}) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}}

Create an empty Onesweep workspace using storage compatible with `KeyV`.

# Parameters

- `::Type{KeyV}`: Vector type used for key storage and derived scratch buffers.

# Returns

A `OnesweepWorkspace` with zero-length buffers for keys, offsets, counters, and permutation storage.
"""
OnesweepWorkspace(::Type{KeyV}) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}}

"""
    initialize_workspace!(ws::OnesweepWorkspace{KeyT}, nelems::Int, ntiles::Int, ::Val{NWorkers}, ::Val{TileSize}) where {KeyT <: Unsigned, NWorkers, TileSize}

Resize and clear the buffers required for Onesweep key sorting.

# Parameters

- `ws`: Workspace whose internal buffers are resized and initialized.
- `nelems`: Number of elements to sort.
- `ntiles`: Number of tiles processed by a radix pass.
- `::Val{NWorkers}`: Compile-time number of workers used for per-worker scratch storage.
- `::Val{TileSize}`: Compile-time tile size used for local rank storage.
"""
function initialize_workspace!(ws :: OnesweepWorkspace{KeyT}, nelems :: Int, ntiles :: Int, :: Val{NWorkers}, :: Val{TileSize}) where {KeyT <: Unsigned, NWorkers, TileSize}
    npass = _npasses(KeyT)    

    resize!(ws.dst, nelems)
    resize!(ws.tile_counter, 1)
    resize!(ws.lookback, 256 * ntiles)

    resize!(ws.bucket_offsets, 256 * npass)
    resize!(ws.prepass_counts, 256 * npass * NWorkers)

    resize!(ws.local_counts,   256 * NWorkers)
    resize!(ws.local_offsets,  256 * NWorkers)
    resize!(ws.global_offsets, 256 * NWorkers)
    resize!(ws.rank_cursors,   256 * NWorkers)
    resize!(ws.local_ranks,   TileSize * NWorkers)

    fill!(ws.tile_counter, zero(UInt32))
    fill!(ws.lookback, zero(UInt32))
    fill!(ws.bucket_offsets, zero(UInt32))
    fill!(ws.prepass_counts, zero(UInt32))

    fill!(ws.local_counts, zero(UInt32))
    fill!(ws.local_offsets, zero(UInt32))
    fill!(ws.global_offsets, zero(UInt32))
    fill!(ws.rank_cursors, zero(UInt32))
    fill!(ws.local_ranks, zero(UInt32))

    return nothing
end

"""
    initialize_perm_indices!(perm::Vector{UInt32}, nelems::Int)

Fill a permutation buffer with 1-based Julia source indices.

# Parameters

- `perm`: Permutation buffer to initialize.
- `nelems`: Number of indices to write.
"""
@inline function initialize_perm_indices!(perm :: Vector{UInt32}, nelems :: Int)
    @inbounds for i in 1:nelems
        perm[i] = UInt32(i)
    end
    return nothing
end

"""
    initialize_perm_workspace!(ws::OnesweepWorkspace{KeyT, KeyV}, nelems::Int, ntiles::Int, ::Val{NWorkers}, ::Val{TileSize}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, NWorkers, TileSize}

Resize and clear the Onesweep workspace for permutation sorting.

# Parameters

- `ws`: Workspace whose key-sorting and permutation buffers are resized and initialized.
- `nelems`: Number of elements to sort.
- `ntiles`: Number of tiles processed by a radix pass.
- `::Val{NWorkers}`: Compile-time number of workers used for per-worker scratch storage.
- `::Val{TileSize}`: Compile-time tile size used for local rank storage.
"""
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

"""
    reset_pass_workspace!(ws::OnesweepWorkspace)

Clear the per-pass tile counter and look-back table.

# Parameters

- `ws`: Workspace whose per-pass buffers are reset.
"""
function reset_pass_workspace!(ws :: OnesweepWorkspace)
    fill!(ws.tile_counter, zero(UInt32))
    fill!(ws.lookback, zero(UInt32))
    return nothing
end
