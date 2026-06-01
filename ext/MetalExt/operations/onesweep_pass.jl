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

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function onesweep_pass_kernel!(
                codes :: KeyV,
                ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV},
                :: Val{TileSize},
                :: Val{Pass},
            ) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, Pass}

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

            # Threadgroup-local equivalent of CUB agent temporary storage.
            #
            # local_counts[bucket]   ~= CUB per-tile bins
            # local_offsets[bucket]  ~= exclusive_digit_prefix
            # rank_cursors[bucket]   ~= temporary cursor for stable rank creation
            # global_offsets[bucket] ~= CUB TempStorage.global_offsets
            # local_ranks[i]         ~= tile-local sorted rank for element i
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            rank_cursors = Metal.MtlThreadGroupArray(UInt32, 256)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            local_ranks = Metal.MtlThreadGroupArray(UInt32, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)

            lane_id = Int(Metal.thread_position_in_threadgroup().x)
            nlanes = Int(Metal.threads_per_threadgroup().x)

            while true
                # Dynamic tile claim, CUDA mental model:
                #
                #   if (threadIdx.x == 0)
                #       shared_tile = atomicAdd(d_tile_counter, 1)
                #   __syncthreads()
                #   tile_id = shared_tile
                #
                # The unit of work is one tile per CUDA block / Metal
                # threadgroup. Only one lane may increment the global counter;
                # the claimed tile id is then broadcast through threadgroup
                # memory so all lanes cooperate on the same tile.
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

                # Clear per-tile bucket storage cooperatively.
                bucket_idx = lane_id
                while bucket_idx <= 256
                    @inbounds begin
                        local_counts[bucket_idx] = zero(UInt32)
                        local_offsets[bucket_idx] = zero(UInt32)
                        rank_cursors[bucket_idx] = zero(UInt32)
                        global_offsets[bucket_idx] = zero(UInt32)
                    end
                    bucket_idx += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # CUB-like local histogram. Every lane scans a block-strided
                # subset of this claimed tile and accumulates into
                # threadgroup/shared memory.
                local_i = lane_id
                while local_i <= tile_len
                    i = rangemin + local_i - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    Metal.atomic_fetch_add_explicit(pointer(local_counts, bucket), UInt32(1))
                    local_i += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # Publish this tile's local counts as PARTIAL lookback entries.
                # This is global memory because later tiles may be processed by
                # different threadgroups.
                bucket_idx = lane_id
                while bucket_idx <= 256
                    @inbounds count = local_counts[bucket_idx]
                    idx = UnsignedRadixSorts._lookback_index(tile_id, bucket_idx)
                    entry = UnsignedRadixSorts._partial_entry(count)
                    Metal.atomic_store_explicit(pointer(lookback, idx), entry)
                    bucket_idx += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # Correctness-first replacement for CUB BlockRadixRank:
                #
                # 1. lane 1 computes exclusive bucket offsets.
                # 2. lane 1 walks the tile in original order to create stable
                #    tile-local ranks.
                #
                # CUB does this with a parallel BlockRadixRank primitive. Keeping
                # this serial for now prevents subtle stability bugs while the
                # surrounding OneSweep protocol is brought up.
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

                # Resolve global offsets through the decoupled lookback table.
                #
                # For each bucket, previous accumulates all same-bucket counts in
                # earlier tiles until a GLOBAL prefix entry is found. The scatter
                # base follows the same formula as the CPU implementation:
                #
                #   global_offsets[bucket] =
                #       bucket_start + previous - local_offsets[bucket]
                bucket_idx = lane_id
                while bucket_idx <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1
                    while prev_tile >= 0
                        idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket_idx)
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
                        local_count = local_counts[bucket_idx]
                        bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket_idx)]
                        global_offsets[bucket_idx] = bucket_start + previous - local_offsets[bucket_idx]
                    end

                    # Upgrade this tile's lookback entry from PARTIAL to GLOBAL.
                    idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket_idx)
                    global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
                    Metal.atomic_store_explicit(pointer(lookback, idx_l), global_entry)

                    bucket_idx += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # Scatter keys. Each lane handles a strided subset of tile items.
                # local_ranks are tile-wide sorted ranks, so adding the bucket's
                # global offset gives the final 1-based Julia output index.
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

        @inline function onesweep_perm_pass_kernel!(
                codes :: KeyV,
                ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV},
                :: Val{TileSize},
                :: Val{Pass},
            ) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, Pass}

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
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            rank_cursors = Metal.MtlThreadGroupArray(UInt32, 256)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            local_ranks = Metal.MtlThreadGroupArray(UInt32, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)

            lane_id = Int(Metal.thread_position_in_threadgroup().x)
            nlanes = Int(Metal.threads_per_threadgroup().x)

            while true
                # One CUDA block / Metal threadgroup claims one tile. All lanes
                # then cooperate on that tile and share the claimed id through
                # threadgroup memory.
                if lane_id == 1
                    @inbounds claimed_tile[1] = Metal.atomic_fetch_add_explicit(pointer(tile_counter, 1), UInt32(1))
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                @inbounds tile_id = Int(claimed_tile[1])
                tile_id < ntiles || break

                rangemin = tile_id * TileSize + 1
                rangemax = min(rangemin + TileSize - 1, nelems)
                tile_len = rangemax - rangemin + 1

                bucket_idx = lane_id
                while bucket_idx <= 256
                    @inbounds begin
                        local_counts[bucket_idx] = zero(UInt32)
                        local_offsets[bucket_idx] = zero(UInt32)
                        rank_cursors[bucket_idx] = zero(UInt32)
                        global_offsets[bucket_idx] = zero(UInt32)
                    end
                    bucket_idx += nlanes
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

                bucket_idx = lane_id
                while bucket_idx <= 256
                    @inbounds count = local_counts[bucket_idx]
                    idx = UnsignedRadixSorts._lookback_index(tile_id, bucket_idx)
                    entry = UnsignedRadixSorts._partial_entry(count)
                    Metal.atomic_store_explicit(pointer(lookback, idx), entry)
                    bucket_idx += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # Correctness-first stable rank construction. This is the part
                # that should later become a real CUB-style BlockRadixRank.
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

                bucket_idx = lane_id
                while bucket_idx <= 256
                    previous = zero(UInt32)

                    prev_tile = tile_id - 1
                    while prev_tile >= 0
                        idx = UnsignedRadixSorts._lookback_index(prev_tile, bucket_idx)
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
                        local_count = local_counts[bucket_idx]
                        bucket_start = bucket_offsets[UnsignedRadixSorts._bucket_offsets_index(Pass, bucket_idx)]
                        global_offsets[bucket_idx] = bucket_start + previous - local_offsets[bucket_idx]
                    end

                    idx_l = UnsignedRadixSorts._lookback_index(tile_id, bucket_idx)
                    global_entry = UnsignedRadixSorts._global_entry(previous + local_count)
                    Metal.atomic_store_explicit(pointer(lookback, idx_l), global_entry)

                    bucket_idx += nlanes
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                # The only semantic difference from key-only sorting: scatter
                # both the key and its original 1-based source index to the same
                # output slot.
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
