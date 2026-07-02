## Local CCCL reference:
##
##   temp/cccl_onesweep/cub/cub/agent/agent_radix_sort_onesweep.cuh
##   detail::radix_sort::AgentRadixSortOnesweep
##
## CCCL Process() reference:
##
##   void Process()
##   {
##       bit_ordered_type keys[ITEMS_PER_THREAD];
##       LoadKeys(block_idx * TILE_ITEMS, keys);
##
##       int ranks[ITEMS_PER_THREAD];
##       int exclusive_digit_prefix[BINS_PER_THREAD];
##       int bins[BINS_PER_THREAD];
##       BlockRadixRankT(s.rank_temp_storage)
##           .RankKeys(keys, ranks, digit_extractor(),
##                     exclusive_digit_prefix,
##                     CountsCallback(*this, bins, keys));
##
##       __syncthreads();
##       ScatterKeysShared(keys, ranks);
##       LoadBinsToOffsetsGlobal(exclusive_digit_prefix);
##       LookbackGlobal(bins);
##       UpdateBinsGlobal(bins, exclusive_digit_prefix);
##
##       __syncthreads();
##       ScatterKeysGlobal();
##       GatherScatterValues(ranks, bool_constant_v<KEYS_ONLY>);
##   }

"""
    onesweep_pass_kernel!(codes::KeyV, ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, ::Val{TileSize}, ::Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}

Execute one OneSweep 8-bit radix pass over keys.

This mirrors CUB `AgentRadixSortOnesweep::Process()` at the stage boundary:
claim `block_idx`, `LoadKeys`, `BlockRadixRankT::RankKeys` with
`CountsCallback`, `ScatterKeysShared`, `LookbackGlobal`, then
`ScatterKeysGlobal`. The CPU helpers keep the same dataflow and use serial
per-worker loops where CUB uses block/warp cooperation.

# Parameters

- `codes`: Key buffer used as the source on odd passes and the destination on even passes.
- `ws`: OneSweep workspace containing buffers, lookback table, counters, offsets, and per-worker temporary storage.
- `::Val{TileSize}`: Compile-time tile size used to partition the input into work tiles.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function onesweep_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
    # Select the active key ping-pong buffers for this radix pass.
    #
    # CCCL equivalent: the dispatch layer passes d_keys_in and d_keys_out into
    # AgentRadixSortOnesweep after choosing the current ping-pong side.
    src, dst = _select_pass_key_buffers(codes, ws, Val(Pass))

    nelems = length(src)
    ntiles = cld(nelems, TileSize)

    # Per-worker equivalent of CUB AgentRadixSortOnesweep::TempStorage_.
    #
    # keys          ~= bit_ordered_type keys[ITEMS_PER_THREAD]
    # rank_cursors  ~= CPU cursor storage corresponding to BlockRadixRankT temp storage
    # local_counts  ~= CountsCallback bins
    # local_offsets ~= exclusive_digit_prefix
    # global_offsets ~= TempStorage_::global_offsets
    # keys_out      ~= TempStorage_::keys_out
    keys = ws.keys
    rank_cursors = ws.rank_cursors
    local_counts = ws.local_counts
    local_offsets = ws.local_offsets
    global_offsets = ws.global_offsets
    local_ranks = ws.local_ranks
    keys_out = ws.keys_out

    while true
        # 1. Agent construction before Process(): claim a tile.
        #
        # CUB's constructor atomically increments d_ctrs, stores block_idx in
        # temporary storage, and Process() handles that claimed tile. CPU
        # workers repeat the same constructor+Process unit until no tile remains.
        tile_id = _claim_next_tile!(ws.tile_counter)
        tile_id < ntiles || break

        # Convert the 0-based CUB-style tile id to Julia's 1-based range.
        rangemin = tile_id * TileSize + 1
        tile_len = min(TileSize, nelems - rangemin + 1)

        # 2. Process(): LoadKeys.
        #
        # Store the tile in the CPU temporary-storage equivalent of CUB's per-thread
        # `keys` register array.
        keys = _load_keys!(src, keys, rangemin, tile_len, Val(TileSize))

        # 3. Process(): BlockRadixRankT::RankKeys with CountsCallback.
        #
        # The helper computes `bins`, publishes LookbackPartial through the
        # CountsCallback path, scans `exclusive_digit_prefix`, and returns
        # CUB-style stable 0-based ranks.
        ranks = _rank_keys_early_counts!(
            keys,
            rank_cursors,
            local_counts,
            local_offsets,
            global_offsets,
            ws.lookback,
            local_ranks,
            tile_id,
            tile_len,
            Val(TileSize),
            Val(Pass),
        )

        # 4. Process(): ScatterKeysShared.
        #
        # Stage keys into s.keys_out[rank] so ScatterKeysGlobal reads the tile
        # in digit-sorted local-rank order.
        _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize))

        # 5. Process(): LoadBinsToOffsetsGlobal -> LookbackGlobal
        #    -> UpdateBinsGlobal.
        #
        # Decoupled lookback combines d_bins, exclusive_digit_prefix, and
        # previous tiles' same-bucket counts into s.global_offsets.
        _resolve_lookback_global_offsets!(
            ws.lookback,
            ws.bucket_offsets,
            local_counts,
            local_offsets,
            global_offsets,
            tile_id,
            Val(Pass),
        )

        # 6. Process(): ScatterKeysGlobal.
        #
        # Sorted s.keys_out entries are written to d_keys_out at
        # sorted_idx + s.global_offsets[Digit(key)]. Keys-only sorting has no
        # value path, so GatherScatterValues is a no-op.
        _scatter_keys_global!(
            keys_out,
            dst,
            global_offsets,
            tile_len,
            Val(TileSize),
            Val(Pass),
        )
    end

    return nothing
end

"""
    onesweep_perm_pass_kernel!(codes::KeyV, ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, ::Val{TileSize}, ::Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}

Execute one OneSweep 8-bit radix pass while scattering keys and associated
permutation indices.

The key path mirrors CUB `ScatterKeysGlobal`. The value path mirrors
`GatherScatterValues`: load values in input-tile order, scatter them through
`values_out` with the retained `ranks`, then write them to the same global
positions as the sorted keys.

# Parameters

- `codes`: Key buffer used as the source on odd passes and the destination on even passes.
- `ws`: OneSweep workspace containing key buffers, permutation buffers, lookback table, counters, offsets, and per-worker temporary storage.
- `::Val{TileSize}`: Compile-time tile size used to partition the input into work tiles.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function onesweep_perm_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
    # Select active key and value ping-pong buffers. For sortperm, values are
    # 1-based source permutation indices.
    src, dst, perm_src, perm_dst = _select_pass_key_value_buffers(codes, ws, Val(Pass))

    nelems = length(src)
    ntiles = cld(nelems, TileSize)

    # Same CUB TempStorage_ mapping as the keys-only pass, plus values_out for
    # GatherScatterValues.
    keys = ws.keys
    rank_cursors = ws.rank_cursors
    local_counts = ws.local_counts
    local_offsets = ws.local_offsets
    global_offsets = ws.global_offsets
    local_ranks = ws.local_ranks
    keys_out = ws.keys_out
    values_out = ws.values_out

    while true
        # 1. Agent construction before Process(): claim a tile.
        tile_id = _claim_next_tile!(ws.tile_counter)
        tile_id < ntiles || break

        # Convert the claimed tile to the current Julia source range.
        rangemin = tile_id * TileSize + 1
        tile_len = min(TileSize, nelems - rangemin + 1)

        # 2. Process(): LoadKeys.
        keys = _load_keys!(src, keys, rangemin, tile_len, Val(TileSize))

        # 3. Process(): RankKeys with CountsCallback and early LookbackPartial.
        ranks = _rank_keys_early_counts!(
            keys,
            rank_cursors,
            local_counts,
            local_offsets,
            global_offsets,
            ws.lookback,
            local_ranks,
            tile_id,
            tile_len,
            Val(TileSize),
            Val(Pass),
        )

        # 4. Process(): ScatterKeysShared.
        _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize))

        # 5. Process(): LoadBinsToOffsetsGlobal -> LookbackGlobal
        #    -> UpdateBinsGlobal.
        _resolve_lookback_global_offsets!(
            ws.lookback,
            ws.bucket_offsets,
            local_counts,
            local_offsets,
            global_offsets,
            tile_id,
            Val(Pass),
        )

        # 6. Process(): ScatterKeysGlobal, then GatherScatterValues.
        _scatter_key_values_global!(
            keys_out,
            values_out,
            keys,
            ranks,
            dst,
            perm_src,
            perm_dst,
            global_offsets,
            rangemin,
            tile_len,
            Val(TileSize),
            Val(Pass),
        )
    end

    return nothing
end
