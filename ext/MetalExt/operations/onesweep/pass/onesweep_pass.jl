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

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function onesweep_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, Pass}
            lookback = ws.lookback
            tile_counter = ws.tile_counter

            # ####################################################
            # Select source/output buffers and bucket-offset buffers for this pass.
            # Pass is a compile-time Val, so this branch is resolved by specialization.
            #
            # CCCL equivalent:
            # AgentRadixSortOnesweep receives d_keys_in and d_keys_out after dispatch selects the active ping-pong buffers.
            # Value pointers are null/ignored for keys-only sorting.
            #
            # CCCL reference code:
            # CCCL source lines: 669-683
            #
            #   AgentRadixSortOnesweep(...,
            #       d_bins_out, d_bins_in,
            #       d_keys_out, d_keys_in,
            #       nullptr, nullptr,
            #       num_items, current_bit, num_bits, decomposer);
            src, dst = _select_pass_key_buffers(codes, ws, Val(Pass))

            nelems = length(src)
            ntiles = cld(nelems, TileSize)
            bucket_offsets = ws.bucket_offsets

            # ####################################################
            # Metal threadgroup-local equivalent of CUB agent temporary storage.
            #
            # local_counts[bucket]   ~= CUB per-tile bins
            # local_offsets[bucket]  ~= exclusive_digit_prefix
            # rank_cursors[bucket]   ~= temporary cursor for stable rank creation
            # simd_offsets[simd,bucket] ~= per-simdgroup histogram/cursor scratch
            # global_offsets[bucket] ~= CUB TempStorage.global_offsets
            # local_ranks[i]         ~= tile-local sorted rank for element i
            #
            # CCCL equivalent:
            # AgentRadixSortOnesweep::TempStorage_ contains rank_temp_storage,
            # keys_out/values_out shared arrays, global_offsets, and block_idx.
            #
            # CCCL reference code:
            # CCCL source lines: 177-190
            #
            #   struct TempStorage_ {
            #       union {
            #           bit_ordered_type keys_out[TILE_ITEMS];
            #           ValueT values_out[TILE_ITEMS];
            #           typename BlockRadixRankT::TempStorage rank_temp_storage;
            #       };
            #       union {
            #           OffsetT global_offsets[RADIX_DIGITS];
            #           PortionOffsetT block_idx;
            #       };
            #   };
            local_counts   = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets  = Metal.MtlThreadGroupArray(UInt32, 256)
            rank_cursors   = Metal.MtlThreadGroupArray(UInt32, 256)
            simd_offsets   = Metal.MtlThreadGroupArray(UInt32, 16 * 256)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            local_ranks    = Metal.MtlThreadGroupArray(UInt32, TileSize)
            claimed_tile   = Metal.MtlThreadGroupArray(UInt32, 1)

            while true
                # ####################################################
                # 1. Agent construction before Process(): claim the tile.
                # CUB obtains block_idx in AgentRadixSortOnesweep constructor, then
                # calls Process() for that claimed tile. This loop repeats the same
                # constructor+Process unit until no tiles remain.
                #
                # CCCL equivalent:
                # AgentRadixSortOnesweep constructor prepares the Process() inputs:
                #
                #   lane 0 atomically increments d_ctrs
                #   the claimed block_idx is shared with the whole block
                #   full_block is computed from block_idx and TILE_ITEMS
                #
                # CCCL reference code:
                # CCCL source lines: 699-708
                #
                #   if (threadIdx.x == 0) {
                #       s.block_idx = atomicAdd(d_ctrs, 1);
                #   }
                #   __syncthreads();
                #   block_idx = s.block_idx;
                #   full_block = (block_idx + 1) * TILE_ITEMS <= num_items;

                # Claim one tile from the global counter.
                tile_id = _claim_next_tile!(tile_counter, claimed_tile)
                tile_id < ntiles || break

                # Convert the 0-based tile id to Julia's 1-based input range.
                rangemin = tile_id * TileSize + 1
                rangemax = min(rangemin + TileSize - 1, nelems)
                tile_len = rangemax - rangemin + 1

                # ####################################################
                # 2. Process(): LoadKeys, then RankKeys count path.
                # CUB loads the tile into per-thread key arrays and enters RankKeys.
                # RankKeys produces bins through CountsCallback; this implementation
                # materializes the same bin counts before building ranks.
                #
                # CCCL equivalent:
                # Process() calls LoadKeys(block_idx * TILE_ITEMS, keys), then
                # BlockRadixRankT::RankKeys(... CountsCallback(...)). The count portion
                # corresponds to the bins later passed to LookbackPartial/LookbackGlobal.
                #
                # CCCL reference code:
                # CCCL source lines: 642-650
                #
                #   bit_ordered_type keys[ITEMS_PER_THREAD];
                #   LoadKeys(block_idx * TILE_ITEMS, keys);
                #   BlockRadixRankT(s.rank_temp_storage)
                #       .RankKeys(keys, ranks, digit_extractor(),
                #                 exclusive_digit_prefix,
                #                 CountsCallback(*this, bins, keys));

                # Clear per-bucket scratch for this tile, then count tile items by radix bucket.
                _clear_tile_storage!(local_counts, local_offsets, rank_cursors, global_offsets)
                _load_keys_and_count_digits!(src, local_counts, rangemin, tile_len, Val(Pass))

                # ####################################################
                # 3. Process(): CountsCallback -> LookbackPartial.
                # During RankKeys, CUB invokes CountsCallback after the per-tile bin
                # counts are known. The callback publishes PARTIAL lookback entries
                # before the pass resolves global offsets.
                #
                # CCCL equivalent:
                # CountsCallback copies RankKeys' other_bins into bins, calls
                # LookbackPartial(bins), and then may try CUB's short-circuit fast path.
                # This implementation mirrors the required PARTIAL publication.
                #
                # CCCL reference code:
                # CCCL source lines: 233-247
                #
                #   AtomicOffsetT& loc = d_lookback[block_idx * RADIX_DIGITS + bin];
                #   PortionOffsetT value = bins[u] | LOOKBACK_PARTIAL_MASK;
                #   ThreadStore(&loc, value);

                # Publish this tile's bucket counts as PARTIAL entries.
                _publish_lookback_partial!(lookback, local_counts, tile_id)

                # ####################################################
                # 4. Process(): RankKeys rank output and ScatterKeysShared equivalent.
                # CUB's same RankKeys call also returns exclusive_digit_prefix and
                # ranks. Process() then uses ScatterKeysShared(keys, ranks) to stage
                # tile-ordered keys in shared memory. This implementation keeps
                # local_offsets/local_ranks and scatters directly from src.
                #
                # CCCL equivalent:
                # BlockRadixRankT::RankKeys returns exclusive_digit_prefix and ranks.
                # The following ScatterKeysShared step is represented here by retaining
                # local_ranks rather than physically staging keys in shared memory.
                #
                # CCCL reference code:
                # CCCL source lines: 645-654
                #
                #   int exclusive_digit_prefix[BINS_PER_THREAD];
                #   int ranks[ITEMS_PER_THREAD];
                #   BlockRadixRankT(s.rank_temp_storage)
                #       .RankKeys(keys, ranks, digit_extractor(),
                #                 exclusive_digit_prefix,
                #                 CountsCallback(*this, bins, keys));
                #   __syncthreads();
                #   ScatterKeysShared(keys, ranks);

                # Build exclusive bucket offsets and stable local ranks.
                _rank_keys_local!(src, local_counts, local_offsets, rank_cursors, local_ranks, simd_offsets, rangemin, tile_len, Val(TileSize), Val(Pass))

                # ####################################################
                # 5. Process(): LoadBinsToOffsetsGlobal -> LookbackGlobal -> UpdateBinsGlobal.
                # This is the decoupled lookback part of OneSweep. It combines the
                # pass-wide bucket base, this tile's exclusive_digit_prefix, and previous
                # tiles' same-bucket counts to produce global_offsets.
                #
                # CCCL equivalent:
                # Process() calls:
                #
                #   LoadBinsToOffsetsGlobal(exclusive_digit_prefix)
                #   LookbackGlobal(bins)
                #   UpdateBinsGlobal(bins, exclusive_digit_prefix)
                #
                # LoadBinsToOffsetsGlobal seeds s.global_offsets[bin] from d_bins_in
                # minus the local exclusive prefix. LookbackGlobal walks backward
                # through d_lookback until it finds a GLOBAL entry, accumulates the
                # payload counts, publishes this block's GLOBAL prefix, and adds the
                # previous-block contribution into s.global_offsets[bin].
                #
                # CCCL reference code:
                # CCCL source lines: 276-311, 461-490
                #
                #   LoadBinsToOffsetsGlobal(offsets):
                #       s.global_offsets[bin] = d_bins_in[bin] - offsets[u];
                #
                #   LookbackGlobal(bins):
                #       PortionOffsetT inc_sum = bins[u];
                #       for (PortionOffsetT block_jdx = block_idx - 1; block_jdx >= 0; --block_jdx) {
                #           value_j = ThreadLoad(&d_lookback[block_jdx * RADIX_DIGITS + bin]);
                #           inc_sum += value_j & LOOKBACK_VALUE_MASK;
                #           if (value_j & LOOKBACK_GLOBAL_MASK) break;
                #       }
                #       ThreadStore(&d_lookback[block_idx * RADIX_DIGITS + bin],
                #                   inc_sum | LOOKBACK_GLOBAL_MASK);
                #       s.global_offsets[bin] += inc_sum - bins[u];

                # Resolve each bucket's global scatter base.
                _resolve_lookback_global_offsets!(lookback, bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, Val(Pass))

                # ####################################################
                # 6. Process(): ScatterKeysGlobal.
                # CUB scatters from s.keys_out using idx + s.global_offsets[Digit(key)].
                # Since this implementation did not stage keys in s.keys_out, it computes the same final index from local_ranks.
                # The write goes directly from src to the pass output.
                #
                # CCCL equivalent:
                # ScatterKeysGlobal computes a per-item global index from the tile item rank plus s.global_offsets[Digit(key)].
                # It then writes the key to d_keys_out.
                # Keys-only Process() stops here; GatherScatterValues is a no-op.
                #
                # CCCL reference code:
                # CCCL source lines: 493-589
                #
                #   int idx = threadIdx.x + u * BLOCK_THREADS;
                #   bit_ordered_type key = s.keys_out[idx];
                #   OffsetT global_idx = idx + s.global_offsets[Digit(key)];
                #   d_keys_out[global_idx] = Twiddle::Out(key, decomposer);

                # Scatter keys to final pass positions.
                _scatter_keys_global!(src, dst, global_offsets, local_ranks, rangemin, tile_len, Val(Pass))
            end

            return nothing
        end

        @inline function onesweep_perm_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, Pass}
            lookback = ws.lookback
            tile_counter = ws.tile_counter

            # ####################################################
            # Select key and value/permutation ping-pong buffers.
            # This is the CUB key-value path; permutation indices are treated
            # as values that follow the key through every pass.
            src, dst, perm_src, perm_dst = _select_pass_key_value_buffers(codes, ws, Val(Pass))

            nelems = length(src)
            ntiles = cld(nelems, TileSize)
            bucket_offsets = ws.bucket_offsets

            # ####################################################
            # Metal threadgroup-local equivalent of CUB agent temporary storage.
            # Sortperm uses the same rank/lookback scratch as key-only sorting;
            # it only adds a second global scatter store for the permutation.
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            rank_cursors = Metal.MtlThreadGroupArray(UInt32, 256)
            simd_offsets = Metal.MtlThreadGroupArray(UInt32, 16 * 256)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            local_ranks = Metal.MtlThreadGroupArray(UInt32, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)

            while true
                # ####################################################
                # 1. Agent construction before Process(): claim the tile.
                tile_id = _claim_next_tile!(tile_counter, claimed_tile)
                tile_id < ntiles || break

                # Convert the 0-based tile id to Julia's 1-based input range.
                rangemin = tile_id * TileSize + 1
                rangemax = min(rangemin + TileSize - 1, nelems)
                tile_len = rangemax - rangemin + 1

                # ####################################################
                # 2. Process(): LoadKeys, then RankKeys count path.
                _clear_tile_storage!(local_counts, local_offsets, rank_cursors, global_offsets)
                _load_keys_and_count_digits!(src, local_counts, rangemin, tile_len, Val(Pass))

                # ####################################################
                # 3. Process(): CountsCallback -> LookbackPartial.
                _publish_lookback_partial!(lookback, local_counts, tile_id)

                # ####################################################
                # 4. Process(): RankKeys rank output and ScatterKeysShared equivalent.
                _rank_keys_local!(src, local_counts, local_offsets, rank_cursors, local_ranks, simd_offsets, rangemin, tile_len, Val(TileSize), Val(Pass))

                # ####################################################
                # 5. Process(): LoadBinsToOffsetsGlobal -> LookbackGlobal -> UpdateBinsGlobal.
                _resolve_lookback_global_offsets!(lookback, bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, Val(Pass))

                # ####################################################
                # 6. Process(): ScatterKeysGlobal, then GatherScatterValues.
                # Write the key and paired permutation value to the same
                # digit-derived global position.
                _scatter_key_values_global!(src, dst, perm_src, perm_dst, global_offsets, local_ranks, rangemin, tile_len, Val(Pass))
            end

            return nothing
        end
    end
end
