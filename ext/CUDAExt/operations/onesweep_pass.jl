## CUB/CUDA vocabulary for this CUDA code:
##
## - CUDA block            ~= CUDA block
## - CUDA thread           ~= CUDA thread
## - threads=(...)         ~= CUDA blockDim.x
## - blocks=(...)          ~= CUDA gridDim.x
## - CuStaticSharedArray   ~= CUDA __shared__ memory
##
## This is a correctness-first, CUB-shaped OneSweep pass. The global dataflow is
## CUB-like:
##
##   claim tile -> local histogram/rank -> publish partial lookback
##              -> resolve global prefix -> scatter
##
## The low-level rank construction is implemented in block_radix_rank.jl.

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function onesweep_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, TileSize, Pass}

            lookback = ws.lookback
            tile_counter = ws.tile_counter

            # Ping-pong source/destination selection. Pass is a compile-time Val,
            # so this branch is resolved by specialization.
            if isodd(Pass)
                src = codes
                dst = ws.dst
            else
                src = ws.dst
                dst = codes
            end

            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            bucket_offsets = ws.bucket_offsets

            # Block-local equivalent of CUB agent temporary storage.
            #
            # local_counts[bucket]   ~= CUB per-tile bins
            # local_offsets[bucket]  ~= exclusive_digit_prefix
            # rank_cursors[bucket]   ~= temporary cursor for stable rank creation
            # global_offsets[bucket] ~= CUB TempStorage.global_offsets
            # local_ranks[i]         ~= tile-local sorted rank for element i
            local_counts = CUDA.CuStaticSharedArray(UInt32, 256)
            local_offsets = CUDA.CuStaticSharedArray(UInt32, 256)
            rank_cursors = CUDA.CuStaticSharedArray(UInt32, 256)
            global_offsets = CUDA.CuStaticSharedArray(UInt32, 256)
            local_ranks = CUDA.CuStaticSharedArray(UInt32, TileSize)
            warp_offsets = CUDA.CuStaticSharedArray(UInt32, 8 * 256)
            claimed_tile = CUDA.CuStaticSharedArray(UInt32, 1)

            thread_id = Int(CUDA.threadIdx().x)
            nthreads = Int(CUDA.blockDim().x)

            while true
                # Dynamic tile claim, CUDA mental model:
                #
                #   if (threadIdx.x == 0)
                #       shared_tile = atomicAdd(d_tile_counter, 1)
                #   __syncthreads()
                #   tile_id = shared_tile
                #
                # The unit of work is one tile per CUDA block. Only one thread may
                # increment the global counter; the claimed tile id is then
                # broadcast through shared memory so all threads cooperate on the
                # same tile.
                if thread_id == 1
                    @inbounds claimed_tile[1] = CUDA.atomic_add!(pointer(tile_counter, 1), UInt32(1))
                end
                CUDA.sync_threads()

                # CUDA.atomic_add! returns the old value, so the stored value is
                # already the 0-based tile id.
                @inbounds tile_id = Int(claimed_tile[1])
                tile_id < ntiles || break

                # Convert 0-based tile id to Julia's 1-based input range.
                rangemin = tile_id * TileSize + 1
                rangemax = min(rangemin + TileSize - 1, nelems)
                tile_len = rangemax - rangemin + 1

                # Clear per-tile bucket storage cooperatively.
                bucket = thread_id
                while bucket <= 256
                    @inbounds begin
                        local_counts[bucket] = zero(UInt32)
                        local_offsets[bucket] = zero(UInt32)
                        rank_cursors[bucket] = zero(UInt32)
                        global_offsets[bucket] = zero(UInt32)
                    end
                    bucket += nthreads
                end
                CUDA.sync_threads()

                # CUB-like local histogram. Every thread scans a block-strided
                # subset of this claimed tile and accumulates into shared memory.
                local_i = thread_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    CUDA.atomic_add!(pointer(local_counts, bucket), UInt32(1))
                    local_i += nthreads
                end
                CUDA.sync_threads()

                # Publish this tile's local counts as PARTIAL lookback entries.
                # This is global memory because later tiles may be processed by
                # different blocks.
                bucket = thread_id
                while bucket <= 256
                    @inbounds count = local_counts[bucket]
                    idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    entry = UnsignedRadixSorts._partial_entry(count)
                    CUDA.atomic_xchg!(pointer(lookback, idx), entry)
                    bucket += nthreads
                end
                CUDA.sync_threads()

                # CUB-style BlockRadixRankMatchEarlyCounts replacement for the
                # old serial rank builder. This computes exclusive_digit_prefix
                # and stable tile-local ranks in parallel.
                block_radix_rank_onesweep!(
                    src,
                    local_counts,
                    local_offsets,
                    rank_cursors,
                    local_ranks,
                    warp_offsets,
                    rangemin,
                    tile_len,
                    Val(TileSize),
                    Val(Pass),
                )

                # Resolve global offsets through the decoupled lookback table.
                #
                # For each bucket, previous accumulates all same-bucket counts in
                # earlier tiles until a GLOBAL prefix entry is found. The scatter
                # base follows the same formula as the CPU implementation:
                #
                #   global_offsets[bucket] =
                #       bucket_start + previous - local_offsets[bucket]
                bucket = thread_id
                while bucket <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1
                    while prev_tile >= 0
                        idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket)
                        entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))

                        # Wait until the previous tile has published either a
                        # PARTIAL count or a GLOBAL prefix.
                        while entry == zero(UInt32)
                            entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))
                        end

                        previous += UnsignedRadixSorts._entry_count(entry)

                        if UnsignedRadixSorts._is_global_entry(entry)
                            break
                        end

                        prev_tile -= 1
                    end

                    @inbounds begin
                        local_count = local_counts[bucket]
                        bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)]
                        global_offsets[bucket] = bucket_start + previous - local_offsets[bucket]
                    end

                    # Upgrade this tile's lookback entry from PARTIAL to GLOBAL.
                    idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
                    CUDA.atomic_xchg!(pointer(lookback, idx_l), global_entry)

                    bucket += nthreads
                end
                CUDA.sync_threads()

                # Scatter keys. Each thread handles a strided subset of tile items.
                # local_ranks are tile-wide sorted ranks, so adding the bucket's
                # global offset gives the final 1-based Julia output index.
                local_i = thread_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds begin
                        bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                        scatter_idx = global_offsets[bucket] + local_ranks[local_i]
                        dst[Int(scatter_idx)] = src[i]
                    end
                    local_i += nthreads
                end
                CUDA.sync_threads()
            end

            return nothing
        end

        @inline function onesweep_perm_pass_kernel!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, TileSize, Pass}

            lookback = ws.lookback
            tile_counter = ws.tile_counter

            # Same ping-pong rule as the key-only pass, plus the matching
            # permutation buffers. The permutation value follows the key through
            # every pass, so the final returned buffer is the stable source order.
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

            bucket_offsets = ws.bucket_offsets

            # CUDA __shared__ equivalent. This is intentionally identical to
            # onesweep_pass_kernel!; sortperm only adds a second scatter store.
            local_counts = CUDA.CuStaticSharedArray(UInt32, 256)
            local_offsets = CUDA.CuStaticSharedArray(UInt32, 256)
            rank_cursors = CUDA.CuStaticSharedArray(UInt32, 256)
            global_offsets = CUDA.CuStaticSharedArray(UInt32, 256)
            local_ranks = CUDA.CuStaticSharedArray(UInt32, TileSize)
            warp_offsets = CUDA.CuStaticSharedArray(UInt32, 8 * 256)
            claimed_tile = CUDA.CuStaticSharedArray(UInt32, 1)

            thread_id = Int(CUDA.threadIdx().x)
            nthreads = Int(CUDA.blockDim().x)

            while true
                # One CUDA block claims one tile. All threads then cooperate on
                # that tile and share the claimed id through shared memory.
                if thread_id == 1
                    @inbounds claimed_tile[1] = CUDA.atomic_add!(pointer(tile_counter, 1), UInt32(1))
                end
                CUDA.sync_threads()

                @inbounds tile_id = Int(claimed_tile[1])
                tile_id < ntiles || break

                rangemin = tile_id * TileSize + 1
                rangemax = min(rangemin + TileSize - 1, nelems)
                tile_len = rangemax - rangemin + 1

                bucket = thread_id
                while bucket <= 256
                    @inbounds begin
                        local_counts[bucket] = zero(UInt32)
                        local_offsets[bucket] = zero(UInt32)
                        rank_cursors[bucket] = zero(UInt32)
                        global_offsets[bucket] = zero(UInt32)
                    end
                    bucket += nthreads
                end
                CUDA.sync_threads()

                local_i = thread_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    CUDA.atomic_add!(pointer(local_counts, bucket), UInt32(1))
                    local_i += nthreads
                end
                CUDA.sync_threads()

                bucket = thread_id
                while bucket <= 256
                    @inbounds count = local_counts[bucket]
                    idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    entry = UnsignedRadixSorts._partial_entry(count)
                    CUDA.atomic_xchg!(pointer(lookback, idx), entry)
                    bucket += nthreads
                end
                CUDA.sync_threads()

                # CUB-style BlockRadixRankMatchEarlyCounts replacement for the
                # old serial rank builder. This computes exclusive_digit_prefix
                # and stable tile-local ranks in parallel.
                block_radix_rank_onesweep!(
                    src,
                    local_counts,
                    local_offsets,
                    rank_cursors,
                    local_ranks,
                    warp_offsets,
                    rangemin,
                    tile_len,
                    Val(TileSize),
                    Val(Pass),
                )

                bucket = thread_id
                while bucket <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1
                    while prev_tile >= 0
                        idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket)
                        entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))

                        while entry == zero(UInt32)
                            entry = CUDA.atomic_add!(pointer(lookback, idx), zero(UInt32))
                        end

                        previous += UnsignedRadixSorts._entry_count(entry)

                        if UnsignedRadixSorts._is_global_entry(entry)
                            break
                        end

                        prev_tile -= 1
                    end

                    @inbounds begin
                        local_count = local_counts[bucket]
                        bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)]
                        global_offsets[bucket] = bucket_start + previous - local_offsets[bucket]
                    end

                    idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket)
                    global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
                    CUDA.atomic_xchg!(pointer(lookback, idx_l), global_entry)

                    bucket += nthreads
                end
                CUDA.sync_threads()

                # The only semantic difference from key-only sorting: scatter
                # both the key and its original 1-based source index to the same
                # output slot.
                local_i = thread_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds begin
                        bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                        scatter_idx = global_offsets[bucket] + local_ranks[local_i]
                        dst[Int(scatter_idx)] = src[i]
                        perm_dst[Int(scatter_idx)] = perm_src[i]
                    end
                    local_i += nthreads
                end
                CUDA.sync_threads()
            end

            return nothing
        end
    end
end
