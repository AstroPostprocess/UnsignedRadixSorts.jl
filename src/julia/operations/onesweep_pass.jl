
## CPU reference for the CUB-shaped OneSweep pass. The global dataflow is
## CUB-like:
##
##   claim tile -> local histogram/rank -> publish partial lookback
##              -> resolve global prefix -> scatter
##
## The CPU path keeps the same logical phases as the GPU kernels, but stores
## temporary state in per-worker workspace arrays instead of threadgroup/shared
## memory.
##
## Local CCCL reference:
##
##   temp/cccl_onesweep/cub/cub/agent/agent_radix_sort_onesweep.cuh
##   detail::radix_sort::AgentRadixSortOnesweep
##
## The high-level CCCL Process() shape is:
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

"""
    onesweep_pass_kernel!(codes::KeyV, ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, ::Val{TileSize}, ::Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}

Execute one OneSweep 8-bit radix pass by claiming tiles, computing local ranks, resolving global bucket offsets, and scattering keys between the active pass buffers.

# Parameters

- `codes`: Key buffer used as the source on odd passes and the destination on even passes.
- `ws`: OneSweep workspace containing the destination buffer, lookback table, counters, offsets, and per-worker scratch buffers.
- `::Val{TileSize}`: Compile-time tile size used to partition the input into work tiles.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function onesweep_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
    lookback     = ws.lookback
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

    # Per-worker equivalent of CUB agent temporary storage.
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
    local_counts   = ws.local_counts
    local_offsets  = ws.local_offsets
    global_offsets = ws.global_offsets
    rank_cursors   = ws.rank_cursors
    local_ranks    = ws.local_ranks

    worker_id = _worker_id()
    tile_base = TileSize * (worker_id - 1)

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
        new = Atomix.@atomic tile_counter[1] += one(UInt32)
        tile_id = Int(new - UInt32(1))
        tile_id < ntiles || break

        # Convert the 0-based tile id to a 1-based Julia range.
        rangemin = tile_id * TileSize + 1
        rangemax = min(rangemin + TileSize - 1, nelems)

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
        @inbounds for bucket in 1:256
            # Get the scratch-buffer index for this worker and bucket.
            idx = _worker_bucket_index(worker_id, bucket)
            local_counts[idx] = zero(UInt32)
        end

        @inbounds for i in rangemin:rangemax
            bucket = _radix_bucket(src[i], Pass)
            idx = _worker_bucket_index(worker_id, bucket)
            local_counts[idx] += one(UInt32)
        end

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
        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            count = local_counts[idx_wb]
            idx = _lookback_index(tile_id, bucket)
            Atomix.@atomic :release lookback[idx] = _partial_entry(count)
        end

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
        running = zero(UInt32)

        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            local_offsets[idx_wb] = running
            running += local_counts[idx_wb]
        end
        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            rank_cursors[idx_wb] = local_offsets[idx_wb]
        end

        @inbounds for i in rangemin:rangemax
            rank_idx = tile_base + i - rangemin + 1
            bucket = _radix_bucket(src[i], Pass)
            idx_wb = _worker_bucket_index(worker_id, bucket)
            rank = rank_cursors[idx_wb]
            local_ranks[rank_idx] = rank
            rank_cursors[idx_wb] = rank + one(UInt32)
        end

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
        @inbounds for bucket in 1:256
            previous = zero(UInt32)

            prev_tile = tile_id - 1
            while prev_tile >= 0
                idx = _lookback_index(prev_tile, bucket)

                entry = Atomix.@atomic :acquire lookback[idx]

                # Wait until the previous tile has published either a PARTIAL
                # count or a GLOBAL prefix.
                while entry == zero(UInt32)
                    entry = Atomix.@atomic :acquire lookback[idx]
                end

                previous += _entry_count(entry)

                if _is_global_entry(entry)
                    break
                end

                prev_tile -= 1
            end

            idx_wb = _worker_bucket_index(worker_id, bucket)
            local_count = local_counts[idx_wb]

            # This tile's actual scatter base for this bucket:
            #
            #   global_offsets[bucket] =
            #       bucket_start + previous - local_offsets[bucket]
            # bucket_start is the pass-wide start of this bucket.
            idx_bo = _bucket_offsets_index(Pass, bucket)
            global_offsets[idx_wb] = bucket_offsets[idx_bo] + previous - local_offsets[idx_wb]

            # Update this tile's lookback entry from PARTIAL to GLOBAL.
            idx_l = _lookback_index(tile_id, bucket)
            Atomix.@atomic :release lookback[idx_l] = _global_entry(previous + local_count)
        end

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
        @inbounds for i in rangemin:rangemax
            rank_idx = tile_base + i - rangemin + 1
            bucket = _radix_bucket(src[i], Pass)
            idx_wb = _worker_bucket_index(worker_id, bucket)
            scatter_idx = global_offsets[idx_wb] + local_ranks[rank_idx]
            dst[Int(scatter_idx)] = src[i]
        end

    end
    
    return nothing
end

"""
    onesweep_perm_pass_kernel!(codes::KeyV, ws::OnesweepWorkspace{KeyT, KeyV, OffsetV}, ::Val{TileSize}, ::Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}

Execute one OneSweep 8-bit radix pass while scattering keys and their associated permutation indices between the active pass buffers.

# Parameters

- `codes`: Key buffer used as the source on odd passes and the destination on even passes.
- `ws`: OneSweep workspace containing key buffers, permutation buffers, lookback table, counters, offsets, and per-worker scratch buffers.
- `::Val{TileSize}`: Compile-time tile size used to partition the input into work tiles.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function onesweep_perm_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyT <: Unsigned, KeyV <: Vector{KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
    lookback     = ws.lookback
    tile_counter = ws.tile_counter

    # ####################################################
    # Select source/output buffers and bucket-offset buffers for this pass.
    # Same ping-pong rule as the key-only pass, plus the matching permutation
    # buffers. The permutation value follows the key through every pass.
    #
    # CCCL equivalent:
    # AgentRadixSortOnesweep receives both key and value input/output pointers.
    # For sortperm, this implementation treats permutation indices as values.
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

    # Per-worker equivalent of CUB agent temporary storage.
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
    # bucket_offsets stores the pass-wide global start position for each radix bucket.
    bucket_offsets = ws.bucket_offsets
    local_counts   = ws.local_counts
    local_offsets  = ws.local_offsets
    global_offsets = ws.global_offsets
    rank_cursors   = ws.rank_cursors
    local_ranks    = ws.local_ranks

    worker_id = _worker_id()
    tile_base = TileSize * (worker_id - 1)

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
        new = Atomix.@atomic tile_counter[1] += one(UInt32)
        tile_id = Int(new - UInt32(1))
        tile_id < ntiles || break

        # Convert the 0-based tile id to a 1-based Julia range.
        rangemin = tile_id * TileSize + 1
        rangemax = min(rangemin + TileSize - 1, nelems)

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
        @inbounds for bucket in 1:256
            idx = _worker_bucket_index(worker_id, bucket)
            local_counts[idx] = zero(UInt32)
        end

        @inbounds for i in rangemin:rangemax
            bucket = _radix_bucket(src[i], Pass)
            idx = _worker_bucket_index(worker_id, bucket)
            local_counts[idx] += one(UInt32)
        end

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
        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            count = local_counts[idx_wb]
            idx = _lookback_index(tile_id, bucket)
            Atomix.@atomic :release lookback[idx] = _partial_entry(count)
        end

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
        running = zero(UInt32)

        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            local_offsets[idx_wb] = running
            running += local_counts[idx_wb]
        end
        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            rank_cursors[idx_wb] = local_offsets[idx_wb]
        end

        @inbounds for i in rangemin:rangemax
            rank_idx = tile_base + i - rangemin + 1
            bucket = _radix_bucket(src[i], Pass)
            idx_wb = _worker_bucket_index(worker_id, bucket)
            rank = rank_cursors[idx_wb]
            local_ranks[rank_idx] = rank
            rank_cursors[idx_wb] = rank + one(UInt32)
        end

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
        @inbounds for bucket in 1:256
            previous = zero(UInt32)

            prev_tile = tile_id - 1
            while prev_tile >= 0
                idx = _lookback_index(prev_tile, bucket)

                entry = Atomix.@atomic :acquire lookback[idx]

                # Wait until the previous tile has published either a PARTIAL
                # count or a GLOBAL prefix.
                while entry == zero(UInt32)
                    entry = Atomix.@atomic :acquire lookback[idx]
                end

                previous += _entry_count(entry)

                if _is_global_entry(entry)
                    break
                end

                prev_tile -= 1
            end

            idx_wb = _worker_bucket_index(worker_id, bucket)
            local_count = local_counts[idx_wb]

            # This tile's actual scatter base for this bucket:
            #
            #   global_offsets[bucket] =
            #       bucket_start + previous - local_offsets[bucket]
            # bucket_start is the pass-wide start of this bucket.
            idx_bo = _bucket_offsets_index(Pass, bucket)
            global_offsets[idx_wb] = bucket_offsets[idx_bo] + previous - local_offsets[idx_wb]

            # Update this tile's lookback entry from PARTIAL to GLOBAL.
            idx_l = _lookback_index(tile_id, bucket)
            Atomix.@atomic :release lookback[idx_l] = _global_entry(previous + local_count)
        end

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
        @inbounds for i in rangemin:rangemax
            rank_idx = tile_base + i - rangemin + 1
            bucket = _radix_bucket(src[i], Pass)
            idx_wb = _worker_bucket_index(worker_id, bucket)
            scatter_idx = global_offsets[idx_wb] + local_ranks[rank_idx]
            dst[Int(scatter_idx)] = src[i]
            perm_dst[Int(scatter_idx)] = perm_src[i]
        end

    end
    
    return nothing
end
