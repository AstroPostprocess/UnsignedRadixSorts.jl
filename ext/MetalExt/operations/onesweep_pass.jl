## CUB/CUDA vocabulary for this Metal code:
##
## - Metal threadgroup      ~= CUDA block
## - Metal thread           ~= CUDA thread
## - threads=(...)          ~= CUDA blockDim.x
## - groups=(...)           ~= CUDA gridDim.x
## - MtlThreadGroupArray    ~= CUDA __shared__ memory
##
## This is a correctness-first, CUB-shaped OneSweep pass. The global dataflow is
## CUB-like:
##
##   claim tile -> local histogram/rank -> publish partial lookback
##              -> resolve global prefix -> scatter
##
## The low-level ranking primitive is not yet CUB BlockRadixRank. Offset/rank
## construction is deliberately serial inside one lane to preserve stable order
## while the surrounding storage and lookback protocol are moved to threadgroup
## memory.
##
## Local CCCL reference:
##
##   temp/cccl_onesweep/cub/cub/agent/agent_radix_sort_onesweep.cuh
##   detail::radix_sort::AgentRadixSortOnesweep
##
## The CCCL Process() shape is:
##
##   LoadKeys
##   BlockRadixRankT::RankKeys(..., exclusive_digit_prefix, CountsCallback)
##   ScatterKeysShared
##   LoadBinsToOffsetsGlobal
##   LookbackGlobal
##   ScatterKeysGlobal
##   GatherScatterValues
##
## Concrete CCCL reference shape:
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
                # 1. Tile claim phase:
                # Dynamically claim one tile id from the global counter.
                # The claimed tile is the work unit corresponding to one CCCL block; the
                # current execution unit processes that tile through all phases below.
                #
                # CCCL equivalent:
                # AgentRadixSortOnesweep constructor:
                #
                #   lane 0 atomically increments d_ctrs
                #   the claimed block_idx is shared with the whole block
                #   full_block is computed from block_idx and TILE_ITEMS
                #
                # CCCL reference code:
                #
                #   if (threadIdx.x == 0) {
                #       s.block_idx = atomicAdd(d_ctrs, 1);
                #   }
                #   __syncthreads();
                #   block_idx = s.block_idx;
                #   full_block = (block_idx + 1) * TILE_ITEMS <= num_items;
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
                # 2. Local histogram phase:
                # Load the claimed tile and compute its local radix histogram.
                # In CCCL the bin counts are produced by RankKeys through
                # CountsCallback; this implementation materializes the count portion
                # before rank construction.
                #
                # CCCL equivalent:
                # Process() first calls LoadKeys(block_idx * TILE_ITEMS, keys),
                # then BlockRadixRankT::RankKeys computes both bins and ranks. This
                # implementation splits the bin-count portion out explicitly.
                #
                # CCCL reference code:
                #
                #   bit_ordered_type keys[ITEMS_PER_THREAD];
                #   LoadKeys(block_idx * TILE_ITEMS, keys);
                #   BlockRadixRankT(s.rank_temp_storage)
                #       .RankKeys(keys, ranks, digit_extractor(),
                #                 exclusive_digit_prefix,
                #                 CountsCallback(*this, bins, keys));
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

                local_i = lane_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    Metal.atomic_fetch_add_explicit(pointer(local_counts, bucket), UInt32(1))
                    local_i += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # ####################################################
                # 3. Partial lookback publish phase:
                # Publish this tile's local counts as PARTIAL lookback entries.
                # Later tiles can consume these entries while resolving decoupled
                # lookback prefixes for each radix bucket.
                #
                # CCCL equivalent:
                # CountsCallback waits for lookback initialization, then calls
                # LookbackPartial(bins), which stores a PARTIAL-masked bin count
                # into d_lookback[block_idx, bin].
                #
                # CCCL reference code:
                #
                #   AtomicOffsetT& loc = d_lookback[block_idx * RADIX_DIGITS + bin];
                #   PortionOffsetT value = bins[u] | LOOKBACK_PARTIAL_MASK;
                #   ThreadStore(&loc, value);
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
                # 4. Local radix rank phase:
                # Compute the tile-local exclusive digit prefix and each item's 0-based
                # rank in bucket-sorted tile order.
                # CCCL produces both values in one BlockRadixRankT::RankKeys call; this
                # implementation stores the prefix in local_offsets and ranks in
                # local_ranks.
                #
                # CCCL equivalent:
                # BlockRadixRankT::RankKeys returns exclusive_digit_prefix and
                # writes ranks[ITEMS_PER_THREAD]. Process() then uses
                # ScatterKeysShared(keys, ranks) before the global scatter. This
                # implementation stores the tile-local ranks in local_ranks and
                # scatters directly from the active source buffer.
                #
                # CCCL reference code:
                #
                #   int exclusive_digit_prefix[BINS_PER_THREAD];
                #   int ranks[ITEMS_PER_THREAD];
                #   BlockRadixRankT(s.rank_temp_storage)
                #       .RankKeys(keys, ranks, digit_extractor(),
                #                 exclusive_digit_prefix,
                #                 CountsCallback(*this, bins, keys));
                #   __syncthreads();
                #   ScatterKeysShared(keys, ranks);
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
                # 5. Global offset lookback phase:
                # Resolve per-bucket global scatter offsets through the decoupled
                # lookback table.
                # For each bucket, accumulate same-bucket counts from earlier tiles until
                # a GLOBAL prefix entry is found, then publish this tile's GLOBAL prefix.
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
                bucket = lane_id
                while bucket <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1
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
                # 6. Global scatter phase:
                # Scatter keys using the bucket base and the 0-based tile-wide rank.
                # Adding the bucket's global offset to the tile-local rank gives the
                # final Julia output index.
                #
                # CCCL equivalent:
                # ScatterKeysGlobal computes a per-item global index from the tile item
                # rank plus s.global_offsets[Digit(key)], then writes to d_keys_out.
                # Keys-only Process() stops here.
                #
                # CCCL reference code:
                #
                #   int idx = threadIdx.x + u * BLOCK_THREADS;
                #   bit_ordered_type key = s.keys_out[idx];
                #   OffsetT global_idx = idx + s.global_offsets[Digit(key)];
                #   d_keys_out[global_idx] = Twiddle::Out(key, decomposer);
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
                # 1. Tile claim phase:
                # Dynamically claim one tile id from the global counter.
                # The claimed tile is the work unit corresponding to one CCCL block; the
                # current execution unit processes that tile through all phases below.
                #
                # CCCL equivalent:
                # AgentRadixSortOnesweep constructor:
                #
                #   lane 0 atomically increments d_ctrs
                #   the claimed block_idx is shared with the whole block
                #   full_block is computed from block_idx and TILE_ITEMS
                #
                # CCCL reference code:
                #
                #   if (threadIdx.x == 0) {
                #       s.block_idx = atomicAdd(d_ctrs, 1);
                #   }
                #   __syncthreads();
                #   block_idx = s.block_idx;
                #   full_block = (block_idx + 1) * TILE_ITEMS <= num_items;
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
                # 2. Local histogram phase:
                # Load the claimed tile and compute its local radix histogram.
                # In CCCL the bin counts are produced by RankKeys through
                # CountsCallback; this implementation materializes the count portion
                # before rank construction.
                #
                # CCCL equivalent:
                # Process() first calls LoadKeys(block_idx * TILE_ITEMS, keys),
                # then BlockRadixRankT::RankKeys computes both bins and ranks. This
                # implementation splits the bin-count portion out explicitly.
                #
                # CCCL reference code:
                #
                #   bit_ordered_type keys[ITEMS_PER_THREAD];
                #   LoadKeys(block_idx * TILE_ITEMS, keys);
                #   BlockRadixRankT(s.rank_temp_storage)
                #       .RankKeys(keys, ranks, digit_extractor(),
                #                 exclusive_digit_prefix,
                #                 CountsCallback(*this, bins, keys));
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

                local_i = lane_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    Metal.atomic_fetch_add_explicit(pointer(local_counts, bucket), UInt32(1))
                    local_i += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # ####################################################
                # 3. Partial lookback publish phase:
                # Publish this tile's local counts as PARTIAL lookback entries.
                # Later tiles can consume these entries while resolving decoupled
                # lookback prefixes for each radix bucket.
                #
                # CCCL equivalent:
                # CountsCallback waits for lookback initialization, then calls
                # LookbackPartial(bins), which stores a PARTIAL-masked bin count
                # into d_lookback[block_idx, bin].
                #
                # CCCL reference code:
                #
                #   AtomicOffsetT& loc = d_lookback[block_idx * RADIX_DIGITS + bin];
                #   PortionOffsetT value = bins[u] | LOOKBACK_PARTIAL_MASK;
                #   ThreadStore(&loc, value);
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
                # 4. Local radix rank phase:
                # Compute the tile-local exclusive digit prefix and each item's 0-based
                # rank in bucket-sorted tile order.
                # CCCL produces both values in one BlockRadixRankT::RankKeys call; this
                # implementation stores the prefix in local_offsets and ranks in
                # local_ranks.
                #
                # CCCL equivalent:
                # BlockRadixRankT::RankKeys returns exclusive_digit_prefix and
                # writes ranks[ITEMS_PER_THREAD]. Process() then uses
                # ScatterKeysShared(keys, ranks) before the global scatter. This
                # implementation stores the tile-local ranks in local_ranks and
                # scatters directly from the active source buffer.
                #
                # CCCL reference code:
                #
                #   int exclusive_digit_prefix[BINS_PER_THREAD];
                #   int ranks[ITEMS_PER_THREAD];
                #   BlockRadixRankT(s.rank_temp_storage)
                #       .RankKeys(keys, ranks, digit_extractor(),
                #                 exclusive_digit_prefix,
                #                 CountsCallback(*this, bins, keys));
                #   __syncthreads();
                #   ScatterKeysShared(keys, ranks);
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
                # 5. Global offset lookback phase:
                # Resolve per-bucket global scatter offsets through the decoupled
                # lookback table.
                # For each bucket, accumulate same-bucket counts from earlier tiles until
                # a GLOBAL prefix entry is found, then publish this tile's GLOBAL prefix.
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
                bucket = lane_id
                while bucket <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1
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
                # 6. Global scatter phase:
                # Scatter keys and permutation values to the same output slot.
                # The only semantic difference from key-only sorting is the additional
                # value scatter that follows the same digit-derived global position.
                #
                # CCCL equivalent:
                # ScatterKeysGlobal writes keys, and GatherScatterValues loads values,
                # scatters them through shared memory by ranks, then writes values to
                # d_values_out at the same digit-derived global positions.
                #
                # CCCL reference code:
                #
                #   ScatterKeysGlobal();
                #   LoadValues(block_idx * TILE_ITEMS, values);
                #   ScatterValuesShared(values, ranks);
                #   ScatterValuesGlobal(digits);
                #   d_values_out[global_idx] = value;
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
