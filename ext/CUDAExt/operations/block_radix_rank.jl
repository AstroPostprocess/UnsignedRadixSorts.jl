## CUDA BlockRadixRank support for OneSweep passes.
##
## This file holds the CUDA-only rank builder used by `onesweep_pass_kernel!`.
## It is intentionally separated from `onesweep_pass.jl` because the rank step
## is the only part of the pass that tries to mirror a CUB block primitive rather
## than ordinary OneSweep control flow.
##
## CUB reference shape:
##
##   cub::BlockRadixRankMatchEarlyCounts<
##       BlockThreads,
##       RadixBits = 8,
##       IsDescending = false,
##       ...
##   >::RankKeys(keys, ranks, digit_extractor, exclusive_digit_prefix, callback)
##
## For this package:
##
## - RadixBits is always 8, so there are always 256 radix buckets.
## - Sorting is ascending only.
## - `local_counts` is already computed by the surrounding OneSweep pass before
##   this helper is called. CUB's early-counts variant can report bucket counts
##   through a callback; here the same count data already exists in shared memory.
## - The helper still has to produce the two outputs that the rest of the pass
##   consumes:
##
##     * `local_offsets[bucket]`: the exclusive digit prefix for each bucket.
##       This is equivalent to CUB's `exclusive_digit_prefix`.
##     * `local_ranks[local_j]`: the stable tile-local rank of each tile item.
##       The later scatter step computes:
##
##           dst[global_offsets[bucket] + local_ranks[local_j]] = src[i]
##
## The implementation below follows the match-early-counts dataflow:
##
##   1. Compute exclusive bucket offsets from `local_counts`.
##   2. Build warp-private bucket histograms for the tile.
##   3. Convert warp-private counts into per-warp bucket starting offsets.
##   4. Rank each warp strip with warp match masks.
##
## CUDA.jl does not currently expose a direct `match_any` intrinsic. To keep the
## semantics equivalent, each lane constructs the same peer mask by shuffling the
## candidate digit from every lane in the warp and comparing it locally. That is
## slower than native `match_any`, but it preserves the CUB rank rule:
##
##   rank = prefix for previous warps in the same bucket
##        + prefix for previous strips in this warp and bucket
##        + number of lower-numbered lanes with the same digit in this strip

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function block_radix_rank_onesweep!(src :: KeyV, local_counts :: OffsetV, local_offsets :: OffsetV, rank_cursors :: OffsetV, local_ranks :: OffsetV,  warp_offsets :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, TileSize, Pass}

            thread_id = Int(CUDA.threadIdx().x)
            nthreads = Int(CUDA.blockDim().x)

            # The sorter launch policy currently caps ThreadsPerBlock at 256.
            # That means at most eight 32-lane CUDA warps per block. The shared
            # `warp_offsets` buffer is therefore sized as 8 * 256 entries by the
            # caller. Keeping this constant here makes the buffer contract
            # explicit at the primitive boundary.
            warp_threads = 32
            max_block_warps = 8

            # CUDA.jl uses 1-based thread ids, matching Julia indexing. CUB's
            # warp formulas are normally written with 0-based warp ids and lane
            # ids, so `lane_in_warp` and `warp_id` are kept 0-based for
            # arithmetic and converted back to 1-based only when indexing Julia
            # arrays.
            warp_id = fld(thread_id - 1, warp_threads)
            lane_in_warp = (thread_id - 1) % warp_threads
            nwarps = cld(nthreads, warp_threads)

            # CUB processes a fixed number of items per thread. This code keeps
            # the same mental model, but the final tile can be partial, so every
            # item access is guarded by `local_j <= tile_len`.
            keys_per_thread = cld(TileSize, nthreads)

            # -----------------------------------------------------------------
            # Stage 1: exclusive digit prefix
            # -----------------------------------------------------------------
            #
            # CUB stores digit counts inside BlockRadixRank temporary storage and
            # uses BlockScan to expose `exclusive_digit_prefix`. The surrounding
            # OneSweep pass already built `local_counts[bucket]`, so we scan that
            # shared array directly.
            #
            # `local_offsets` is used first as the inclusive-scan working array.
            # `rank_cursors` is a second 256-entry scratch array used to avoid
            # read-after-write hazards between Hillis-Steele scan stages.
            bucket = thread_id
            while bucket <= 256
                @inbounds local_offsets[bucket] = local_counts[bucket]
                bucket += nthreads
            end
            CUDA.sync_threads()

            offset = 1
            while offset < 256
                bucket = thread_id
                while bucket <= 256
                    addend = bucket > offset ? local_offsets[bucket - offset] : zero(UInt32)
                    @inbounds rank_cursors[bucket] = local_offsets[bucket] + addend
                    bucket += nthreads
                end
                CUDA.sync_threads()

                bucket = thread_id
                while bucket <= 256
                    @inbounds local_offsets[bucket] = rank_cursors[bucket]
                    bucket += nthreads
                end
                CUDA.sync_threads()

                offset <<= 1
            end

            # Convert the inclusive scan into CUB's exclusive digit prefix:
            #
            #   local_offsets[1]      = 0
            #   local_offsets[bucket] = count(bucket 1) + ... + count(bucket - 1)
            bucket = thread_id
            while bucket <= 256
                @inbounds rank_cursors[bucket] = bucket == 1 ? zero(UInt32) : local_offsets[bucket - 1]
                bucket += nthreads
            end
            CUDA.sync_threads()

            bucket = thread_id
            while bucket <= 256
                @inbounds local_offsets[bucket] = rank_cursors[bucket]
                bucket += nthreads
            end
            CUDA.sync_threads()

            # -----------------------------------------------------------------
            # Stage 2: warp-private histograms
            # -----------------------------------------------------------------
            #
            # CUB's match-early-counts implementation stores a histogram for each
            # warp and each digit. We use `warp_offsets[warp, bucket]` for that
            # temporary histogram first. Later the same storage becomes the
            # per-warp bucket cursor used by the rank loop.
            idx = thread_id
            while idx <= max_block_warps * 256
                @inbounds warp_offsets[idx] = zero(UInt32)
                idx += nthreads
            end
            CUDA.sync_threads()

            item = 0
            while item < keys_per_thread
                # Warp-striped tile layout:
                #
                #   local_j = warp section base
                #           + strip within the warp section
                #           + lane within the warp
                #
                # This is the arrangement assumed by CUB's match-based rank
                # algorithms. The output rank is still written to
                # `local_ranks[local_j]`, so the later scatter loop can continue
                # iterating over the natural tile order.
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

            # -----------------------------------------------------------------
            # Stage 3: per-warp digit offsets
            # -----------------------------------------------------------------
            #
            # For each bucket, convert:
            #
            #   warp_offsets[warp, bucket] = count in that warp
            #
            # into:
            #
            #   warp_offsets[warp, bucket] =
            #       local_offsets[bucket] + count of same-bucket keys in all
            #       earlier warps
            #
            # This gives each warp a stable starting cursor for every bucket.
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

            # -----------------------------------------------------------------
            # Stage 4: warp match ranking
            # -----------------------------------------------------------------
            #
            # CUB's native path uses match-any to obtain a peer mask of lanes in
            # the current warp strip that have the same digit. CUDA.jl exposes
            # shuffle and ballot primitives but not match-any directly, so each
            # lane reconstructs its peer mask by comparing its digit with every
            # lane's shuffled digit.
            #
            # `lane_mask_lt` has bits set for lanes lower than the current lane.
            # Counting `peer_mask & lane_mask_lt` gives the stable in-strip rank
            # contribution for this lane.
            lane_mask_lt = CUDA.lanemask(<)
            warp_lane_id = Int(CUDA.laneid())
            full_mask = CUDA.FULL_MASK

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

                # Use the first participating lane as the digit leader. The
                # leader advances this warp's bucket cursor exactly once for the
                # whole peer group, matching CUB's "leader lane updates cursor,
                # then shuffles the result" pattern.
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

                # Keep the warp-level cursor update and next strip separated.
                # CUB uses `__syncwarp()` around the equivalent match/cursor
                # update sequence.
                CUDA.sync_warp(full_mask)
                item += 1
            end
            CUDA.sync_threads()

            return nothing
        end
    end
end
