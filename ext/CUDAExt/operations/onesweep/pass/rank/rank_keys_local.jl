## CUDA BlockRadixRank-style local rank helper for OneSweep.
##
##   Upstream CCCL: cub/cub/block/block_radix_rank.cuh
##   cub::BlockRadixRankMatchEarlyCounts::RankKeys
##
## This helper is the CUDA implementation of the OneSweep pass step:
##
##   BlockRadixRankT(s.rank_temp_storage)
##       .RankKeys(keys, ranks, digit_extractor(),
##                 exclusive_digit_prefix,
##                 CountsCallback(*this, bins, keys));
##
## At this point `local_counts` already contains the CountCallback-style bins
## for the tile. `_rank_keys_local!` performs the BRR work that turns those
## bins into exclusive digit prefixes and stable tile-local ranks.
##
## Outputs:
## - `local_offsets[bucket]` ~= CUB `exclusive_digit_prefix`
## - `local_ranks[local_j]` ~= stable tile-local rank

"""
    _rank_keys_local!(src, local_counts, local_offsets, rank_cursors, local_ranks, warp_offsets, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Compute CUDA BlockRadixRank-style local prefixes and ranks for a tile.

`local_counts` is the per-tile digit histogram produced before
`CountsCallback -> LookbackPartial`. This helper scans those counts into
`local_offsets`, builds per-warp digit cursors in `warp_offsets`, and writes a
stable tile-local rank for every valid tile item into `local_ranks`.

CUB parallel: `BlockRadixRankMatchEarlyCounts::RankKeys` with early counts
already materialized.

# Parameters

- `src`: Active source key buffer for this pass.
- `local_counts`: Shared-memory tile digit counts, equivalent to CUB `bins`.
- `local_offsets`: Shared-memory exclusive digit prefixes written in-place.
- `rank_cursors`: Shared-memory scratch used by the prefix scan.
- `local_ranks`: Shared-memory stable tile-local ranks written in-place.
- `warp_offsets`: Shared-memory per-warp histogram/cursor scratch.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _rank_keys_local! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _rank_keys_local!(src :: KeyV, local_counts :: OffsetV, local_offsets :: OffsetV, rank_cursors :: OffsetV, local_ranks :: OffsetV, warp_offsets :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, TileSize, Pass}
            thread_id = Int(CUDA.threadIdx().x)
            nthreads = Int(CUDA.blockDim().x)

            # ####################################################
            # RankKeys() setup: CUB prepares BlockRadixRank temp storage and
            # warp/lane metadata before computing digit prefixes and ranks.
            #
            # CUB reference:
            # - block_radix_rank.cuh lines 897-924: warp/lane setup
            # - block_radix_rank.cuh lines 1137-1144: RankKeys entry setup
            warp_threads = Int(CUDA.warpsize())
            max_block_warps = cld(256, warp_threads)
            full_mask = CUDA.FULL_MASK

            # Keep CUB-style 0-based warp/lane ids for rank arithmetic.
            # CUDA.jl's laneid is 1-based, so keep both forms available.
            warp_id = fld(thread_id - 1, warp_threads)
            lane_in_warp = (thread_id - 1) % warp_threads
            warp_lane_id = Int(CUDA.laneid())
            nwarps = cld(nthreads, warp_threads)

            # Partial tiles are guarded by `local_j <= tile_len`.
            keys_per_thread = cld(TileSize, nthreads)

            # ####################################################
            # 1. RankKeys(): bins -> exclusive_digit_prefix.
            #
            # CUB reference:
            # - block_radix_rank.cuh lines 1127-1130
            #
            # RankKeys uses BlockScan to convert per-tile `bins` into
            # `exclusive_digit_prefix`. Here `local_counts` already contains
            # those bins from the earlier count path, so this helper scans the
            # 256 bucket counts into `local_offsets`.
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
            # CUB reference:
            # - block_radix_rank.cuh lines 951-1014
            #
            # BRR computes per-warp histograms after the block-level digit
            # prefixes are known. `warp_offsets` is first used as
            # warp_offsets[warp, bucket] = count, so each warp can later receive
            # a bucket-local cursor range.
            idx = thread_id
            while idx <= max_block_warps * 256
                @inbounds warp_offsets[idx] = zero(UInt32)
                idx += nthreads
            end
            CUDA.sync_threads()

            item = 0
            while item < keys_per_thread
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
            # CUB reference:
            # - block_radix_rank.cuh lines 1016-1053, 1132
            #
            # Convert per-warp bucket counts into per-warp cursors. Each bucket
            # starts from `exclusive_digit_prefix[bucket]`; each warp receives
            # the running cursor for that bucket, then the running value is
            # advanced by that warp's count.
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
            # CUB reference:
            # - block_radix_rank.cuh lines 1091-1115, 1134
            #
            # BRR assigns each item a stable tile-local rank by grouping lanes
            # with the same digit using WARP_MATCH_ANY. CUDA.jl does not expose
            # match_any, so the helper rebuilds the peer mask with shuffles.
            # The peer-group leader advances the warp's bucket cursor once,
            # then each lane adds its peer-local prefix.
            lane_mask_lt = CUDA.lanemask(<)

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

                while src_lane <= warp_threads
                    peer_digit = CUDA.shfl_sync(full_mask, digit, src_lane)
                    peer_valid = CUDA.shfl_sync(full_mask, valid_flag, src_lane)

                    if valid && peer_valid == UInt32(1) && peer_digit == digit
                        peer_mask |= UInt32(1) << UInt32(src_lane - 1)
                    end

                    src_lane += 1
                end

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
