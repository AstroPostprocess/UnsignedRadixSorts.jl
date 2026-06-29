## CUDA BlockRadixRank helper for OneSweep.
##
##   temp/cccl_onesweep_sources/cccl/cub/cub/block/block_radix_rank.cuh
##   cub::BlockRadixRankMatchEarlyCounts::RankKeys
##
## Outputs:
## - `local_offsets[bucket]` ~= CUB `exclusive_digit_prefix`
## - `local_ranks[local_j]` ~= stable tile-local rank

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function block_radix_rank_onesweep!(src :: KeyV, local_counts :: OffsetV, local_offsets :: OffsetV, rank_cursors :: OffsetV, local_ranks :: OffsetV,  warp_offsets :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, TileSize, Pass}

            thread_id = Int(CUDA.threadIdx().x)
            nthreads = Int(CUDA.blockDim().x)

            # CUB lines 897-924, 1137-1144: temp storage and warp/lane setup.
            warp_threads = Int(CUDA.warpsize())
            max_block_warps = cld(256, warp_threads)
            full_mask = CUDA.FULL_MASK

            # Keep CUB-style 0-based warp/lane ids for rank arithmetic.
            warp_id = fld(thread_id - 1, warp_threads)
            lane_in_warp = (thread_id - 1) % warp_threads
            warp_lane_id = Int(CUDA.laneid())
            nwarps = cld(nthreads, warp_threads)

            # Partial tiles are guarded by `local_j <= tile_len`.
            keys_per_thread = cld(TileSize, nthreads)

            # ####################################################
            # 1. RankKeys(): bins -> exclusive_digit_prefix.
            #
            # CUB lines 1127-1130 use BlockScan. local_counts is already
            # available, so scan buckets with warp shuffles plus warp totals.
            scan_base_idx = 256

            if thread_id == 1
                @inbounds rank_cursors[scan_base_idx] = zero(UInt32)
            end
            CUDA.sync_threads()

            chunk_start = 0
            while chunk_start < 256
                bucket = chunk_start + thread_id
                valid_bucket = bucket <= 256
                count = zero(UInt32)

                if valid_bucket
                    @inbounds count = local_counts[bucket]
                end

                inclusive = count

                # Inclusive scan within each warp.
                offset = 1
                while offset < warp_threads
                    addend = CUDA.shfl_up_sync(full_mask, inclusive, offset)
                    if warp_lane_id > offset
                        inclusive += addend
                    end
                    offset <<= 1
                end

                if warp_lane_id == warp_threads
                    @inbounds rank_cursors[warp_id + 1] = inclusive
                end
                CUDA.sync_threads()

                warp_total = zero(UInt32)

                if warp_lane_id <= nwarps
                    @inbounds warp_total = rank_cursors[warp_lane_id]
                end

                warp_prefix = warp_total

                # First warp scans the warp totals.
                if warp_id == 0
                    offset = 1
                    while offset < warp_threads
                        addend = CUDA.shfl_up_sync(full_mask, warp_prefix, offset)
                        if warp_lane_id > offset
                            warp_prefix += addend
                        end
                        offset <<= 1
                    end

                    if warp_lane_id <= nwarps
                        @inbounds rank_cursors[warp_lane_id] = warp_prefix - warp_total
                    end
                end
                CUDA.sync_threads()

                @inbounds scan_base = rank_cursors[scan_base_idx]
                @inbounds warp_prefix = rank_cursors[warp_id + 1]

                if valid_bucket
                    @inbounds local_offsets[bucket] = scan_base + warp_prefix + inclusive - count
                end

                if bucket == min(chunk_start + nthreads, 256)
                    @inbounds rank_cursors[scan_base_idx] = scan_base + warp_prefix + inclusive
                end
                CUDA.sync_threads()

                chunk_start += nthreads
            end
            CUDA.sync_threads()

            # ####################################################
            # 2. RankKeys(): ComputeHistogramsWarp(keys).
            #
            # CUB lines 951-1014. warp_offsets is first used as per-warp counts.

            # Clear per-warp bucket counters.
            idx = thread_id
            while idx <= max_block_warps * 256
                @inbounds warp_offsets[idx] = zero(UInt32)
                idx += nthreads
            end
            CUDA.sync_threads()

            # Count tile items by warp and bucket.
            item = 0
            while item < keys_per_thread
                # CUB-style warp-striped local index.
                local_j = warp_id * warp_threads * keys_per_thread + item * warp_threads + lane_in_warp + 1

                if warp_id < nwarps && local_j <= tile_len
                    i = rangemin + local_j - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    idx = warp_id * 256 + bucket
                    CUDA.atomic_add!(pointer(warp_offsets, idx), UInt32(1))
                end

                item += 1
            end
            CUDA.sync_threads()

            # ####################################################
            # 3. RankKeys(): ComputeOffsetsWarpDownsweep(exclusive_digit_prefix).
            #
            # CUB lines 1016-1053, 1132. Convert per-warp counts into
            # per-warp cursors seeded by exclusive_digit_prefix.
            bucket = thread_id
            while bucket <= 256
                @inbounds running = local_offsets[bucket]
                warp = 0

                while warp < nwarps
                    idx = warp * 256 + bucket
                    @inbounds count = warp_offsets[idx]
                    @inbounds warp_offsets[idx] = running
                    running += count
                    warp += 1
                end

                bucket += nthreads
            end
            CUDA.sync_threads()

            # ####################################################
            # 4. RankKeys(): ComputeRanksItem(keys, ranks, WARP_MATCH_ANY).
            #
            # CUB lines 1091-1115, 1134 use match_any. CUDA.jl lacks it, so we
            # rebuild the peer mask with shuffles.
            lane_mask_lt = CUDA.lanemask(<)

            # Rank each warp-striped item within its bucket.
            item = 0
            while item < keys_per_thread
                local_j = warp_id * warp_threads * keys_per_thread + item * warp_threads + lane_in_warp + 1
                valid = warp_id < nwarps && local_j <= tile_len

                digit = UInt32(0)
                bucket = 1

                if valid
                    i = rangemin + local_j - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    digit = UInt32(bucket - 1)
                end

                valid_flag = valid ? UInt32(1) : UInt32(0)
                peer_mask = UInt32(0)
                src_lane = 1

                # Rebuild match_any by comparing against every lane's digit.
                while src_lane <= warp_threads
                    peer_digit = CUDA.shfl_sync(full_mask, digit, src_lane)
                    peer_valid = CUDA.shfl_sync(full_mask, valid_flag, src_lane)

                    if valid && peer_valid == UInt32(1) && peer_digit == digit
                        peer_mask |= UInt32(1) << UInt32(src_lane - 1)
                    end

                    src_lane += 1
                end

                # The leader advances this warp's bucket cursor once per peer group.
                leader_lane = valid ? CUDA.ffs(peer_mask) : Int32(1)
                digit_count = UInt32(CUDA.popc(peer_mask))
                peer_digit_prefix = UInt32(CUDA.popc(peer_mask & lane_mask_lt))
                warp_prefix = zero(UInt32)

                if valid && warp_lane_id == Int(leader_lane)
                    idx = warp_id * 256 + bucket
                    warp_prefix = CUDA.atomic_add!(pointer(warp_offsets, idx), digit_count)
                end

                warp_prefix = CUDA.shfl_sync(full_mask, warp_prefix, leader_lane)

                if valid
                    @inbounds local_ranks[local_j] = warp_prefix + peer_digit_prefix
                end

                CUDA.sync_warp(full_mask)
                item += 1
            end
            CUDA.sync_threads()

            return nothing
        end
    end
end
