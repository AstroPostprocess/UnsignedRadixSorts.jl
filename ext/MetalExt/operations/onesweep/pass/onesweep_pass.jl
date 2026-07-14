using StaticArrays: MVector

## Legacy fused Metal port of CCCL's AgentRadixSortOnesweep::Process().
##
## This kernel is retained as a diagnostic/reference implementation. Public
## Metal sorting uses `onesweep_direct_pass.jl`, whose separate rank, prefix,
## and scatter dispatches do not require cross-threadgroup forward progress.
##
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
##
## This file intentionally keeps the CUDAExt/CUB stage order and helper names.
## The only algorithmic deviation in the current Metal baseline is the staging
## between RankKeys and ScatterKeysGlobal:
##
## - CUDAExt keeps `ranks` in per-thread register storage, writes
##   `keys_out[rank] = key`, and then scatters the sorted shared tile.
## - MetalExt writes ranks to threadgroup memory indexed by the original
##   tile-local input position, skips the `keys_out` shared permutation, and
##   re-reads source keys during global scatter.
##
## The re-read staging is deliberate. It avoids the Metal.jl path that was found
## unstable for `MVector ranks -> keys_out` shared permutation while preserving
## the OneSweep invariant
##
##   output = global_bucket_base + previous_tile_count + stable_tile_rank.

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function onesweep_pass_kernel!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            NSimdgroups = ThreadsPerGroup ÷ 32
            ItemsPerThread = cld(TileSize, ThreadsPerGroup)

            # Select the active key ping-pong buffers for this radix pass.
            #
            # CCCL equivalent:
            # The dispatch layer passes d_keys_in and d_keys_out into
            # AgentRadixSortOnesweep after choosing the current ping-pong side.
            src, dst = _select_pass_key_buffers(codes, ws, Val(Pass))

            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            # Metal threadgroup-memory equivalent of CUB agent temporary storage.
            #
            # simd_offsets   ~= BlockRadixRank per-SIMD digit counts/cursors
            # local_counts   ~= CountsCallback bins
            # local_offsets  ~= exclusive_digit_prefix
            # scan_scratch   ~= RankKeys scan storage for SIMD totals
            # match_scratch  ~= Metal replacement storage for WARP_MATCH_ANY
            # global_offsets ~= TempStorage_::global_offsets, reused as
            #                   RankKeys scan storage before lookback fills it
            # claimed_tile   ~= TempStorage_::block_idx
            # ranks          ~= Metal-safe tile-local ranks, indexed by the
            #                   original tile-local input position
            simd_offsets = Metal.MtlThreadGroupArray(UInt32, NSimdgroups * 256)
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            scan_scratch = Metal.MtlThreadGroupArray(UInt32, NSimdgroups)
            match_scratch = Metal.MtlThreadGroupArray(UInt32, ThreadsPerGroup)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)
            ranks = Metal.MtlThreadGroupArray(UInt32, TileSize)

            # Threadgroup memory with UInt64 keys and ThreadsPerGroup=256 is
            # kept below Apple's 32 KiB limit by not allocating CUB-style
            # keys_out/value_out payload storage in this baseline.
            while true
                # 1. Agent construction before Process(): claim a tile.
                #
                # CUB's constructor atomically increments d_ctrs, stores block_idx
                # in shared memory, and Process() handles that claimed tile. This
                # kernel repeats the same constructor+Process unit until the pass
                # counter runs past ntiles.
                tile_id = _claim_next_tile!(ws.tile_counter, claimed_tile)
                tile_id < ntiles || break

                # Convert the 0-based CUB-style tile id to Julia's 1-based range.
                # The last tile may be partial; every downstream helper guards
                # work with tile_len.
                rangemin = tile_id * TileSize + 1
                tile_len = min(TileSize, nelems - rangemin + 1)

                # 2. Process(): LoadKeys.
                #
                # Keys are loaded into per-thread register storage using the same
                # SIMD-striped ownership that the rank helper uses later. Unlike
                # CUDAExt, these cached keys are not used by a shared permutation
                # stage; global scatter re-reads the source buffer for stability.
                _clear_rank_storage!(simd_offsets)
                keys = zero(MVector{ItemsPerThread, $KeyT})
                _load_keys!(
                    keys,
                    src,
                    rangemin,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerGroup),
                )

                # 3. Process(): BlockRadixRankT::RankKeys with CountsCallback.
                #
                # This helper computes per-tile bins, publishes LookbackPartial
                # as soon as those bins are known, scans bins into
                # exclusive_digit_prefix, and writes stable tile-local ranks.
                #
                # Metal-specific staging: ranks are written to threadgroup memory
                # at `ranks[local_j]`, where `local_j` is the original tile-local
                # input position. This replaces CUDAExt's per-thread `MVector`
                # ranks and is the anchor used by the re-read scatter stage.
                _rank_keys_early_counts!(
                    ranks,
                    keys,
                    simd_offsets,
                    local_counts,
                    local_offsets,
                    scan_scratch,
                    match_scratch,
                    ws.lookback,
                    tile_id,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerGroup),
                    Val(Pass),
                )

                # 4. Process(): ScatterKeysShared.
                #
                # CUDAExt performs `keys_out[rank] = key` here so later global
                # scatter can walk a digit-sorted shared tile. MetalExt deliberately
                # skips this stage: the `MVector ranks -> keys_out` permutation was
                # the unstable path on Metal. The helper name is still kept in
                # scatter_keys_global.jl for layout parity with CUDAExt.

                # 5. Process(): LoadBinsToOffsetsGlobal -> LookbackGlobal
                #    -> UpdateBinsGlobal.
                #
                # The decoupled lookback path combines the pass-wide bucket base,
                # this tile's exclusive_digit_prefix, and previous tiles'
                # same-bucket counts into global_offsets.
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
                # Metal-safe re-read staging: each thread walks original tile order,
                # re-loads `src[rangemin + local_i - 1]`, and combines the saved
                # `ranks[local_i]` with the lookback-resolved global bucket base.
                # This costs one extra sequential source read but avoids the shared
                # key permutation that is used by CUDAExt.
                _scatter_keys_global!(
                    src,
                    dst,
                    global_offsets,
                    ranks,
                    rangemin,
                    tile_len,
                    Val(Pass),
                )
            end

            return nothing
        end

        @inline function onesweep_perm_pass_kernel!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            NSimdgroups = ThreadsPerGroup ÷ 32
            ItemsPerThread = cld(TileSize, ThreadsPerGroup)

            # Select the active key and value ping-pong buffers. For sortperm,
            # the "values" are 1-based source permutation indices.
            src, dst, perm_src, perm_dst = _select_pass_key_value_buffers(codes, ws, Val(Pass))

            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            # Same CUB agent temporary-storage mapping as the keys-only pass.
            # MetalExt keeps ranks in threadgroup memory and avoids CUB's
            # keys_out/values_out shared permutation payloads.
            simd_offsets = Metal.MtlThreadGroupArray(UInt32, NSimdgroups * 256)
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            scan_scratch = Metal.MtlThreadGroupArray(UInt32, NSimdgroups)
            match_scratch = Metal.MtlThreadGroupArray(UInt32, ThreadsPerGroup)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)
            ranks = Metal.MtlThreadGroupArray(UInt32, TileSize)

            # The sortperm pass uses the same re-read staging as keys-only
            # sorting.  Both keys and permutation values are read in original
            # tile order and scattered through the saved ranks.
            while true
                # 1. Agent construction before Process(): claim a tile.
                tile_id = _claim_next_tile!(ws.tile_counter, claimed_tile)
                tile_id < ntiles || break

                # Convert the claimed tile to the current Julia source range.
                rangemin = tile_id * TileSize + 1
                tile_len = min(TileSize, nelems - rangemin + 1)

                # 2. Process(): LoadKeys.
                _clear_rank_storage!(simd_offsets)
                keys = zero(MVector{ItemsPerThread, $KeyT})
                _load_keys!(
                    keys,
                    src,
                    rangemin,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerGroup),
                )

                # 3. Process(): RankKeys with CountsCallback and early
                # LookbackPartial publication.
                _rank_keys_early_counts!(
                    ranks,
                    keys,
                    simd_offsets,
                    local_counts,
                    local_offsets,
                    scan_scratch,
                    match_scratch,
                    ws.lookback,
                    tile_id,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerGroup),
                    Val(Pass),
                )

                # 4. Process(): ScatterKeysShared.
                # Deliberately skipped for the same reason as the keys-only pass.

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
                #
                # CUDAExt stages keys and values through sorted shared payloads.
                # MetalExt re-reads keys and permutation values from the original
                # source positions, uses `ranks[local_i]` for the same stable tile
                # order, and writes both outputs to identical scatter positions.
                _scatter_key_values_global!(
                    src,
                    dst,
                    perm_src,
                    perm_dst,
                    global_offsets,
                    ranks,
                    rangemin,
                    tile_len,
                    Val(Pass),
                )
            end

            return nothing
        end
    end
end
