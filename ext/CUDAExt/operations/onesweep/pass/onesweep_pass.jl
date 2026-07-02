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

@inline function _onesweep_pass_scratch_words(:: Val{ThreadsPerBlock}) where {ThreadsPerBlock}
    NWarps = div(ThreadsPerBlock, 32)
    return NWarps * 256 + 256 + 256 + 256 + 1
end

@inline _align_up_bytes(offset :: Int, alignment :: Int) = cld(offset, alignment) * alignment

@inline function _onesweep_pass_payload_offset(:: Type{KeyT}, :: Val{ThreadsPerBlock}) where {KeyT, ThreadsPerBlock}
    scratch_bytes = _onesweep_pass_scratch_words(Val(ThreadsPerBlock)) * sizeof(UInt32)
    return _align_up_bytes(scratch_bytes, max(sizeof(KeyT), sizeof(UInt32)))
end

@inline function _onesweep_pass_shmem_bytes(:: Type{KeyT}, :: Val{TileSize}, :: Val{ThreadsPerBlock}) where {KeyT, TileSize, ThreadsPerBlock}
    payload_bytes = TileSize * max(sizeof(KeyT), sizeof(UInt32))
    return _onesweep_pass_payload_offset(KeyT, Val(ThreadsPerBlock)) + payload_bytes
end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function onesweep_pass_kernel!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize}, :: Val{ThreadsPerBlock}, :: Val{Pass}) where {CodeV <: CuDeviceVector{$KeyT}, WorkspaceKeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, TileSize, ThreadsPerBlock, Pass}
            # Select the active key ping-pong buffers for this radix pass.
            #
            # CCCL equivalent:
            # The dispatch layer passes d_keys_in and d_keys_out into
            # AgentRadixSortOnesweep after choosing the current ping-pong side.
            src, dst = _select_pass_key_buffers(codes, ws, Val(Pass))

            nelems = length(src)
            ntiles = cld(nelems, TileSize)
            NWarps = div(ThreadsPerBlock, 32)

            # CUDA dynamic shared-memory equivalent of CUB agent temporary storage.
            #
            # CUDA.jl static shared arrays of the same element type and length
            # can alias in this pattern, so keep all UInt32 temporary storage in one
            # manually partitioned dynamic buffer.
            # warp_offsets   ~= BlockRadixRank per-warp digit counts/cursors
            # local_counts   ~= CountsCallback bins
            # local_offsets  ~= exclusive_digit_prefix
            # global_offsets ~= TempStorage_::global_offsets, reused as
            #                   RankKeys scan storage before lookback fills it
            # claimed_tile   ~= TempStorage_::block_idx
            # keys_out       ~= TempStorage_::keys_out
            shared_offset = 0
            warp_offsets = CUDA.CuDynamicSharedArray(UInt32, NWarps * 256, shared_offset)
            shared_offset += NWarps * 256 * sizeof(UInt32)
            local_counts = CUDA.CuDynamicSharedArray(UInt32, 256, shared_offset)
            shared_offset += 256 * sizeof(UInt32)
            local_offsets = CUDA.CuDynamicSharedArray(UInt32, 256, shared_offset)
            shared_offset += 256 * sizeof(UInt32)
            global_offsets = CUDA.CuDynamicSharedArray(UInt32, 256, shared_offset)
            shared_offset += 256 * sizeof(UInt32)
            claimed_tile = CUDA.CuDynamicSharedArray(UInt32, 1, shared_offset)
            keys_out = CUDA.CuDynamicSharedArray($KeyT, TileSize, _onesweep_pass_payload_offset($KeyT, Val(ThreadsPerBlock)))

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
                # same warp-striped ownership that the rank helper uses later.
                _clear_rank_storage!(warp_offsets, Val(NWarps))
                keys = _load_keys!(src, rangemin, tile_len, Val(TileSize), Val(ThreadsPerBlock))

                # 3. Process(): BlockRadixRankT::RankKeys with CountsCallback.
                #
                # This helper computes per-tile bins, publishes LookbackPartial
                # as soon as those bins are known, scans bins into
                # exclusive_digit_prefix, and returns stable tile-local ranks.
                ranks = _rank_keys_early_counts!(
                    keys,
                    warp_offsets,
                    local_counts,
                    local_offsets,
                    global_offsets,
                    ws.lookback,
                    tile_id,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerBlock),
                    Val(Pass),
                )

                # 4. Process(): ScatterKeysShared.
                #
                # CUB stages keys into s.keys_out by rank so global scatter can
                # read the tile in digit-sorted order.
                _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize), Val(ThreadsPerBlock))

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
                # Sorted shared keys are written to d_keys_out at
                # sorted_idx + s.global_offsets[Digit(key)]. Keys-only sorting
                # has no value path, so GatherScatterValues is a no-op.
                _scatter_keys_global!(
                    keys_out,
                    dst,
                    global_offsets,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerBlock),
                    Val(Pass),
                )
            end

            return nothing
        end

        @inline function onesweep_perm_pass_kernel!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize}, :: Val{ThreadsPerBlock}, :: Val{Pass}) where {CodeV <: CuDeviceVector{$KeyT}, WorkspaceKeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, TileSize, ThreadsPerBlock, Pass}
            # Select the active key and value ping-pong buffers. For sortperm,
            # the "values" are 1-based source permutation indices.
            src, dst, perm_src, perm_dst = _select_pass_key_value_buffers(codes, ws, Val(Pass))

            nelems = length(src)
            ntiles = cld(nelems, TileSize)
            NWarps = div(ThreadsPerBlock, 32)

            # Same CUB agent temporary-storage mapping as the keys-only pass,
            # with keys_out later reused as UInt32 value staging storage inside
            # GatherScatterValues.
            shared_offset = 0
            warp_offsets = CUDA.CuDynamicSharedArray(UInt32, NWarps * 256, shared_offset)
            shared_offset += NWarps * 256 * sizeof(UInt32)
            local_counts = CUDA.CuDynamicSharedArray(UInt32, 256, shared_offset)
            shared_offset += 256 * sizeof(UInt32)
            local_offsets = CUDA.CuDynamicSharedArray(UInt32, 256, shared_offset)
            shared_offset += 256 * sizeof(UInt32)
            # Reuse this slice as RankKeys scan storage before lookback
            # overwrites it with final per-bucket scatter bases.
            global_offsets = CUDA.CuDynamicSharedArray(UInt32, 256, shared_offset)
            shared_offset += 256 * sizeof(UInt32)
            claimed_tile = CUDA.CuDynamicSharedArray(UInt32, 1, shared_offset)
            payload_offset = _onesweep_pass_payload_offset($KeyT, Val(ThreadsPerBlock))
            keys_out = CUDA.CuDynamicSharedArray($KeyT, TileSize, payload_offset)
            values_out = CUDA.CuDynamicSharedArray(UInt32, TileSize, payload_offset)

            while true
                # 1. Agent construction before Process(): claim a tile.
                tile_id = _claim_next_tile!(ws.tile_counter, claimed_tile)
                tile_id < ntiles || break

                # Convert the claimed tile to the current Julia source range.
                rangemin = tile_id * TileSize + 1
                tile_len = min(TileSize, nelems - rangemin + 1)

                # 2. Process(): LoadKeys.
                _clear_rank_storage!(warp_offsets, Val(NWarps))
                keys = _load_keys!(src, rangemin, tile_len, Val(TileSize), Val(ThreadsPerBlock))

                # 3. Process(): RankKeys with CountsCallback and early
                # LookbackPartial publication.
                ranks = _rank_keys_early_counts!(
                    keys,
                    warp_offsets,
                    local_counts,
                    local_offsets,
                    global_offsets,
                    ws.lookback,
                    tile_id,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerBlock),
                    Val(Pass),
                )

                # 4. Process(): ScatterKeysShared.
                _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize), Val(ThreadsPerBlock))

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
                # Keys are written first. The value path then loads permutation
                # values in original tile order, scatters them through shared
                # memory by the retained ranks, and writes them to the same
                # global positions as their keys.
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
                    Val(ThreadsPerBlock),
                    Val(Pass),
                )
            end

            return nothing
        end
    end
end
