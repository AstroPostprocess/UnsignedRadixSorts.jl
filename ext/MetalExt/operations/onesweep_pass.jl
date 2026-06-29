## Local CCCL reference:
##
##   temp/cccl_onesweep_sources/cccl/cub/cub/agent/agent_radix_sort_onesweep.cuh
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
            # AgentRadixSortOnesweep receives d_keys_in and d_keys_out after the
            # dispatch layer has selected the active ping-pong buffers for this radix
            # pass. Value pointers are null/ignored for keys-only sorting.
            #
            # CCCL reference code:
            # CCCL source lines: 669-683
            #
            #   AgentRadixSortOnesweep(...,
            #       d_bins_out, d_bins_in,
            #       d_keys_out, d_keys_in,
            #       nullptr, nullptr,
            #       num_items, current_bit, num_bits, decomposer);
            if isodd(Pass)
                src = codes
                dst = ws.dst
            else
                src = ws.dst
                dst = codes
            end

            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            # bucket_offsets stores the pass-wide global start position for each radix bucket.
            bucket_offsets = ws.bucket_offsets

            # Threadgroup-local equivalent of CUB agent temporary storage.
            #
            # local_counts[bucket]   ~= CUB per-tile bins
            # local_offsets[bucket]  ~= exclusive_digit_prefix
            # rank_cursors[bucket]   ~= temporary cursor for stable rank creation
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
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            local_ranks    = Metal.MtlThreadGroupArray(UInt32, TileSize)
            claimed_tile   = Metal.MtlThreadGroupArray(UInt32, 1)

            lane_id = Int(Metal.thread_position_in_threadgroup().x)
            nlanes  = Int(Metal.threads_per_threadgroup().x)

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
                if lane_id == 1
                    @inbounds claimed_tile[1] = Metal.atomic_fetch_add_explicit(pointer(tile_counter, 1), UInt32(1))
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # Metal.atomic_fetch_add_explicit returns the old value, so the
                # stored value is already the 0-based tile id.
                @inbounds tile_id = Int(claimed_tile[1])
                tile_id < ntiles || break

                # Convert 0-based tile id to Julia's 1-based input range.
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

                # Clear per-bucket scratch for this tile.
                bucket = lane_id
                while bucket <= 256
                    @inbounds begin
                        local_counts[bucket] = zero(UInt32)
                        local_offsets[bucket] = zero(UInt32)
                        rank_cursors[bucket] = zero(UInt32)
                        global_offsets[bucket] = zero(UInt32)
                    end
                    bucket += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # Count tile items by radix bucket.
                local_i = lane_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    Metal.atomic_fetch_add_explicit(pointer(local_counts, bucket), UInt32(1))
                    local_i += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

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
                bucket = lane_id
                while bucket <= 256
                    @inbounds count = local_counts[bucket]
                    idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    entry = UnsignedRadixSorts._partial_entry(count)
                    Metal.atomic_store_explicit(pointer(lookback, idx), entry)
                    bucket += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

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
                if lane_id == 1
                    running = zero(UInt32)
                    @inbounds for bucket in 1:256
                        local_offsets[bucket] = running
                        rank_cursors[bucket] = running
                        running += local_counts[bucket]
                    end

                    @inbounds for local_j in 1:tile_len
                        i = rangemin + local_j - 1
                        bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                        rank = rank_cursors[bucket]
                        local_ranks[local_j] = rank
                        rank_cursors[bucket] = rank + one(UInt32)
                    end
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

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
                bucket = lane_id
                while bucket <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1

                    # Walk backward until a GLOBAL prefix is found.
                    while prev_tile >= 0
                        idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket)
                        entry = Metal.atomic_load_explicit(pointer(lookback, idx))

                        # Wait until the previous tile has published either a
                        # PARTIAL count or a GLOBAL prefix.
                        while entry == zero(UInt32)
                            entry = Metal.atomic_load_explicit(pointer(lookback, idx))
                        end

                        previous += UnsignedRadixSorts._entry_count(entry)

                        if UnsignedRadixSorts._is_global_entry(entry)
                            break
                        end

                        prev_tile -= 1
                    end

                    @inbounds begin
                        local_count = local_counts[bucket]
                        # bucket_start is the pass-wide start of this bucket.
                        bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)]
                        global_offsets[bucket] = bucket_start + previous - local_offsets[bucket]
                    end

                    # Upgrade this tile's lookback entry from PARTIAL to GLOBAL.
                    idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
                    Metal.atomic_store_explicit(pointer(lookback, idx_l), global_entry)

                    bucket += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # ####################################################
                # 6. Process(): ScatterKeysGlobal.
                # CUB scatters from s.keys_out using idx + s.global_offsets[Digit(key)].
                # Since this implementation did not stage keys in s.keys_out, it
                # computes the same final index from local_ranks and writes directly
                # from src.
                #
                # CCCL equivalent:
                # ScatterKeysGlobal computes a per-item global index from the tile item
                # rank plus s.global_offsets[Digit(key)], then writes to d_keys_out.
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
                local_i = lane_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds begin
                        bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                        scatter_idx = global_offsets[bucket] + local_ranks[local_i]
                        dst[Int(scatter_idx)] = src[i]
                    end
                    local_i += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            end

            return nothing
        end

        @inline function onesweep_perm_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, Pass}

            lookback = ws.lookback
            tile_counter = ws.tile_counter

            # ####################################################
            # Select source/output buffers and bucket-offset buffers for this pass.
            # Same ping-pong rule as the key-only pass, plus the matching
            # permutation buffers. The permutation value follows the key through
            # every pass, so the final returned buffer is the stable source order.
            #
            # CCCL equivalent:
            # AgentRadixSortOnesweep receives both key and value input/output
            # pointers. For sortperm, this implementation treats permutation
            # indices as values.
            #
            # CCCL reference code:
            # CCCL source lines: 669-683
            #
            #   AgentRadixSortOnesweep(...,
            #       d_bins_out, d_bins_in,
            #       d_keys_out, d_keys_in,
            #       d_values_out, d_values_in,
            #       num_items, current_bit, num_bits, decomposer);
            if isodd(Pass)
                src = codes
                dst = ws.dst
                perm_src = ws.perms[1]
                perm_dst = ws.perms[2]
            else
                src = ws.dst
                dst = codes
                perm_src = ws.perms[2]
                perm_dst = ws.perms[1]
            end

            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            # bucket_offsets stores the pass-wide global start position for each radix bucket.
            bucket_offsets = ws.bucket_offsets

            # CUDA __shared__ equivalent. This is intentionally identical to
            # onesweep_pass_kernel!; sortperm only adds a second scatter store.
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
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            rank_cursors = Metal.MtlThreadGroupArray(UInt32, 256)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            local_ranks = Metal.MtlThreadGroupArray(UInt32, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)

            lane_id = Int(Metal.thread_position_in_threadgroup().x)
            nlanes = Int(Metal.threads_per_threadgroup().x)

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
                if lane_id == 1
                    @inbounds claimed_tile[1] = Metal.atomic_fetch_add_explicit(pointer(tile_counter, 1), UInt32(1))
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                @inbounds tile_id = Int(claimed_tile[1])
                tile_id < ntiles || break

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

                # Clear per-bucket scratch for this tile.
                bucket = lane_id
                while bucket <= 256
                    @inbounds begin
                        local_counts[bucket] = zero(UInt32)
                        local_offsets[bucket] = zero(UInt32)
                        rank_cursors[bucket] = zero(UInt32)
                        global_offsets[bucket] = zero(UInt32)
                    end
                    bucket += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # Count tile items by radix bucket.
                local_i = lane_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    Metal.atomic_fetch_add_explicit(pointer(local_counts, bucket), UInt32(1))
                    local_i += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

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
                bucket = lane_id
                while bucket <= 256
                    @inbounds count = local_counts[bucket]
                    idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    entry = UnsignedRadixSorts._partial_entry(count)
                    Metal.atomic_store_explicit(pointer(lookback, idx), entry)
                    bucket += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

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
                if lane_id == 1
                    running = zero(UInt32)
                    @inbounds for bucket in 1:256
                        local_offsets[bucket] = running
                        rank_cursors[bucket] = running
                        running += local_counts[bucket]
                    end

                    @inbounds for local_j in 1:tile_len
                        i = rangemin + local_j - 1
                        bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                        rank = rank_cursors[bucket]
                        local_ranks[local_j] = rank
                        rank_cursors[bucket] = rank + one(UInt32)
                    end
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

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
                bucket = lane_id
                while bucket <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1

                    # Walk backward until a GLOBAL prefix is found.
                    while prev_tile >= 0
                        idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket)
                        entry = Metal.atomic_load_explicit(pointer(lookback, idx))

                        while entry == zero(UInt32)
                            entry = Metal.atomic_load_explicit(pointer(lookback, idx))
                        end

                        previous += UnsignedRadixSorts._entry_count(entry)

                        if UnsignedRadixSorts._is_global_entry(entry)
                            break
                        end

                        prev_tile -= 1
                    end

                    @inbounds begin
                        local_count = local_counts[bucket]
                        # bucket_start is the pass-wide start of this bucket.
                        bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)]
                        global_offsets[bucket] = bucket_start + previous - local_offsets[bucket]
                    end

                    idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
                    Metal.atomic_store_explicit(pointer(lookback, idx_l), global_entry)

                    bucket += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # ####################################################
                # 6. Process(): ScatterKeysGlobal, then GatherScatterValues.
                # CUB scatters keys first, then for key-value sorting gathers values,
                # scatters them through shared memory by ranks, and writes them to the
                # same digit-derived positions. This implementation writes the
                # permutation value next to the key at the computed final index.
                #
                # CCCL equivalent:
                # ScatterKeysGlobal writes keys. GatherScatterValues loads values,
                # scatters them by ranks through shared memory, then ScatterValuesGlobal
                # writes d_values_out at the same digit-derived global positions.
                #
                # CCCL reference code:
                # CCCL source lines: 614-630, 663
                #
                #   ScatterKeysGlobal();
                #   LoadValues(block_idx * TILE_ITEMS, values);
                #   ScatterValuesShared(values, ranks);
                #   ScatterValuesGlobal(digits);
                #   d_values_out[global_idx] = value;

                # Scatter keys and permutation indices to final pass positions.
                local_i = lane_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds begin
                        bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                        scatter_idx = global_offsets[bucket] + local_ranks[local_i]
                        dst[Int(scatter_idx)] = src[i]
                        perm_dst[Int(scatter_idx)] = perm_src[i]
                    end
                    local_i += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            end

            return nothing
        end
    end
end
