using StaticArrays: MVector

## Metal port of CCCL's AgentRadixSortOnesweep::Process().
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
## The stage order, per-thread cached keys/ranks, early counts publication,
## shared key staging, lookback, and global scatter follow the CUDA version.
## Metal-specific substitutions are limited to backend intrinsics. The default
## Metal policy uses TileSize=2048 to fit the 32 KiB threadgroup-memory budget.

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
            # global_offsets ~= TempStorage_::global_offsets
            # claimed_tile   ~= TempStorage_::block_idx
            # keys_out       ~= TempStorage_::keys_out
            simd_offsets = Metal.MtlThreadGroupArray(UInt32, NSimdgroups * 256)
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            scan_scratch = Metal.MtlThreadGroupArray(UInt32, NSimdgroups)
            match_scratch = Metal.MtlThreadGroupArray(UInt32, ThreadsPerGroup)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            keys_out = Metal.MtlThreadGroupArray($KeyT, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)

            # Threadgroup memory with UInt64 keys and ThreadsPerGroup=256 is
            # 28,708 bytes, below Metal's 32 KiB limit.
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
                # Keys are loaded once into per-thread register storage using the
                # same SIMD-striped ownership that the rank helper uses later.
                _clear_rank_storage!(simd_offsets)
                keys = zero(MVector{ItemsPerThread, $KeyT})
                ranks = zero(MVector{ItemsPerThread, UInt32})
                _load_keys!(keys, src, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup))

                # 3. Process(): BlockRadixRankT::RankKeys with CountsCallback.
                #
                # This helper computes per-tile bins, publishes LookbackPartial
                # as soon as those bins are known, scans bins into
                # exclusive_digit_prefix, and writes stable tile-local ranks.
                _rank_keys_early_counts!(ranks, keys, simd_offsets, local_counts, local_offsets, scan_scratch, match_scratch, ws.lookback, tile_id, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))

                # 4. Process(): ScatterKeysShared.
                #
                # CUB stages keys into s.keys_out by rank so global scatter can
                # read the tile in digit-sorted order.
                _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize), Val(ThreadsPerGroup))

                # 5. Process(): LoadBinsToOffsetsGlobal -> LookbackGlobal
                #    -> UpdateBinsGlobal.
                #
                # The decoupled lookback path combines the pass-wide bucket base,
                # this tile's exclusive_digit_prefix, and previous tiles'
                # same-bucket counts into global_offsets.
                _resolve_lookback_global_offsets!(ws.lookback, ws.bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, Val(Pass))

                # 6. Process(): ScatterKeysGlobal.
                #
                # Sorted threadgroup keys are written to d_keys_out at
                # sorted_idx + s.global_offsets[Digit(key)]. Keys-only sorting
                # has no value path, so GatherScatterValues is a no-op.
                _scatter_keys_global!(keys_out, dst, global_offsets, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))
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
            # UInt8/UInt16 allocate a separate UInt32 values_out staging array
            # because one key slot is narrower than one permutation value.
            # UInt32/UInt64 reuse the key staging array after all sorted keys
            # have been read.
            simd_offsets = Metal.MtlThreadGroupArray(UInt32, NSimdgroups * 256)
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            scan_scratch = Metal.MtlThreadGroupArray(UInt32, NSimdgroups)
            match_scratch = Metal.MtlThreadGroupArray(UInt32, ThreadsPerGroup)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            keys_out = Metal.MtlThreadGroupArray($KeyT, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)
            $(sizeof(KeyT) < sizeof(UInt32) ? :(values_out = Metal.MtlThreadGroupArray(UInt32, TileSize)) : :(nothing))

            # The UInt64 sortperm path has the same 28,708-byte threadgroup
            # footprint as key-only sorting because it reuses per-thread keys
            # as the value staging source.
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
                ranks = zero(MVector{ItemsPerThread, UInt32})
                _load_keys!(keys, src, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup))

                # 3. Process(): RankKeys with CountsCallback and early
                # LookbackPartial publication.
                _rank_keys_early_counts!(ranks, keys, simd_offsets, local_counts, local_offsets, scan_scratch, match_scratch, ws.lookback, tile_id, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))

                # 4. Process(): ScatterKeysShared.
                _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize), Val(ThreadsPerGroup))

                # 5. Process(): LoadBinsToOffsetsGlobal -> LookbackGlobal
                #    -> UpdateBinsGlobal.
                _resolve_lookback_global_offsets!(ws.lookback, ws.bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, Val(Pass))

                # 6. Process(): ScatterKeysGlobal, then GatherScatterValues.
                #
                # Keys are written first. The value path then loads permutation
                # values in original tile order, scatters them through
                # threadgroup memory by the retained ranks, and writes them to
                # the same global positions as their keys.
                $(sizeof(KeyT) < sizeof(UInt32) ?
                    :(_scatter_key_values_global!(keys_out, values_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))) :
                    :(_scatter_key_values_global!(keys_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))))
            end

            return nothing
        end
    end
end
