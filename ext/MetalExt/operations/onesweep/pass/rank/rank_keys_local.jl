## Metal BlockRadixRank-style local rank helper for OneSweep.
##
##   Upstream CCCL: cub/cub/block/block_radix_rank.cuh
##   cub::BlockRadixRankMatchEarlyCounts::RankKeys
##
## This helper is the Metal implementation of the OneSweep pass step:
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
    _rank_keys_local!(src, local_counts, local_offsets, rank_cursors, local_ranks, simd_offsets, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Compute Metal BlockRadixRank-style local prefixes and ranks for a tile.

`local_counts` is the per-tile digit histogram produced before
`CountsCallback -> LookbackPartial`. This helper scans those counts into
`local_offsets`, builds per-simdgroup digit cursors in `simd_offsets`, and
writes a stable tile-local rank for every valid tile item into `local_ranks`.

CUB parallel: `BlockRadixRankMatchEarlyCounts::RankKeys` with early counts
already materialized.

# Parameters

- `src`: Active Metal source key buffer for this pass.
- `local_counts`: Threadgroup-memory tile digit counts, equivalent to CUB `bins`.
- `local_offsets`: Threadgroup-memory exclusive digit prefixes written in-place.
- `rank_cursors`: Threadgroup-memory scratch used by the prefix scan and rank broadcast.
- `local_ranks`: Threadgroup-memory tile-local ranks written in-place.
- `simd_offsets`: Threadgroup-memory per-simdgroup histogram/cursor scratch.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _rank_keys_local! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _rank_keys_local!(src :: KeyV, local_counts :: SharedV, local_offsets :: SharedV, rank_cursors :: SharedV, local_ranks :: SharedV, simd_offsets :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, SharedV <: MtlDeviceVector{UInt32}, TileSize, Pass}
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)

            # ####################################################
            # RankKeys() setup: CUB prepares BlockRadixRank temp storage and
            # warp/lane metadata before computing digit prefixes and ranks.
            #
            # Metal's simdgroup is the execution unit that corresponds to the
            # CUDA warp used by the CUB implementation.
            simd_threads = Int(Metal.threads_per_simdgroup())
            max_threadgroup_simdgroups = cld(256, simd_threads)

            # Keep CUB-style 0-based simdgroup/lane ids for rank arithmetic.
            # Metal.jl's simdgroup and lane ids are 1-based.
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1
            simd_lane_id = lane_in_simd + 1
            nsimdgroups = cld(nthreads, simd_threads)

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
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

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
                while offset < simd_threads
                    addend = Metal.simd_shuffle_up(inclusive, offset)
                    if simd_lane_id > offset
                        inclusive += addend
                    end
                    offset <<= 1
                end

                if simd_lane_id == simd_threads
                    @inbounds rank_cursors[simd_id + 1] = inclusive
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                simd_total = zero(UInt32)

                if simd_lane_id <= nsimdgroups
                    @inbounds simd_total = rank_cursors[simd_lane_id]
                end

                simd_prefix = simd_total

                if simd_id == 0
                    offset = 1
                    while offset < simd_threads
                        addend = Metal.simd_shuffle_up(simd_prefix, offset)
                        if simd_lane_id > offset
                            simd_prefix += addend
                        end
                        offset <<= 1
                    end

                    if simd_lane_id <= nsimdgroups
                        @inbounds rank_cursors[simd_lane_id] = simd_prefix - simd_total
                    end
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                @inbounds scan_base = rank_cursors[scan_base_idx]
                @inbounds simd_prefix = rank_cursors[simd_id + 1]

                if valid_bucket
                    @inbounds local_offsets[bucket] = scan_base + simd_prefix + inclusive - count
                end

                if bucket == min(chunk_start + nthreads, 256)
                    @inbounds rank_cursors[scan_base_idx] = scan_base + simd_prefix + inclusive
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                chunk_start += nthreads
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # ####################################################
            # 2. RankKeys(): ComputeHistogramsWarp(keys).
            #
            # CUB reference:
            # - block_radix_rank.cuh lines 951-1014
            #
            # BRR computes per-warp histograms after the block-level digit
            # prefixes are known. `simd_offsets` is first used as
            # simd_offsets[simdgroup, bucket] = count, so each simdgroup can
            # later receive a bucket-local cursor range.
            idx = thread_id
            while idx <= max_threadgroup_simdgroups * 256
                @inbounds simd_offsets[idx] = zero(UInt32)
                idx += nthreads
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            item = 0
            while item < keys_per_thread
                local_j = simd_id * simd_threads * keys_per_thread + item * simd_threads + lane_in_simd + 1

                if simd_id < nsimdgroups && local_j <= tile_len
                    i = rangemin + local_j - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    idx = simd_id * 256 + bucket
                    Metal.atomic_fetch_add_explicit(pointer(simd_offsets, idx), UInt32(1))
                end

                item += 1
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # ####################################################
            # 3. RankKeys(): ComputeOffsetsWarpDownsweep(exclusive_digit_prefix).
            #
            # CUB reference:
            # - block_radix_rank.cuh lines 1016-1053, 1132
            #
            # Convert per-simdgroup bucket counts into per-simdgroup cursors.
            # Each bucket starts from `exclusive_digit_prefix[bucket]`; each
            # simdgroup receives the running cursor for that bucket, then the
            # running value is advanced by that simdgroup's count.
            bucket = thread_id
            while bucket <= 256
                @inbounds running = local_offsets[bucket]
                simdgroup = 0

                while simdgroup < nsimdgroups
                    idx = simdgroup * 256 + bucket
                    @inbounds count = simd_offsets[idx]
                    @inbounds simd_offsets[idx] = running
                    running += count
                    simdgroup += 1
                end

                bucket += nthreads
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # ####################################################
            # 4. RankKeys(): ComputeRanksItem(keys, ranks, WARP_MATCH_ANY).
            #
            # CUB reference:
            # - block_radix_rank.cuh lines 1091-1115, 1134
            #
            # BRR assigns each item a stable tile-local rank by grouping lanes
            # with the same digit using WARP_MATCH_ANY. Metal.jl does not expose
            # match_any or arbitrary source-lane shuffle, so this helper rebuilds
            # the peer mask through threadgroup scratch. The peer-group leader
            # advances the simdgroup's bucket cursor once, then each lane adds
            # its peer-local prefix.
            item = 0
            while item < keys_per_thread
                local_j = simd_id * simd_threads * keys_per_thread + item * simd_threads + lane_in_simd + 1
                valid = simd_id < nsimdgroups && local_j <= tile_len

                digit = UInt32(0)
                bucket = 1

                if valid
                    i = rangemin + local_j - 1
                    @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                    digit = UInt32(bucket - 1)
                end

                @inbounds rank_cursors[thread_id] = digit
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                peer_mask = UInt32(0)
                src_lane = 1

                while src_lane <= simd_threads
                    peer_thread = simd_id * simd_threads + src_lane
                    peer_local_j = simd_id * simd_threads * keys_per_thread + item * simd_threads + src_lane

                    if valid && peer_thread <= nthreads && peer_local_j <= tile_len
                        @inbounds peer_digit = rank_cursors[peer_thread]

                        if peer_digit == digit
                            peer_mask |= UInt32(1) << UInt32(src_lane - 1)
                        end
                    end

                    src_lane += 1
                end

                leader_lane = Int32(1)
                digit_count = zero(UInt32)
                peer_digit_prefix = zero(UInt32)
                scan_lane = 1

                while scan_lane <= simd_threads
                    bit = UInt32(1) << UInt32(scan_lane - 1)

                    if (peer_mask & bit) != zero(UInt32)
                        if digit_count == zero(UInt32)
                            leader_lane = Int32(scan_lane)
                        end

                        digit_count += one(UInt32)

                        if scan_lane < simd_lane_id
                            peer_digit_prefix += one(UInt32)
                        end
                    end

                    scan_lane += 1
                end

                if valid && simd_lane_id == Int(leader_lane)
                    idx = simd_id * 256 + bucket
                    simd_prefix = Metal.atomic_fetch_add_explicit(pointer(simd_offsets, idx), digit_count)
                    @inbounds rank_cursors[thread_id] = simd_prefix
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                if valid
                    leader_thread = simd_id * simd_threads + Int(leader_lane)
                    @inbounds simd_prefix = rank_cursors[leader_thread]
                    @inbounds local_ranks[local_j] = simd_prefix + peer_digit_prefix
                end
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                item += 1
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            return nothing
        end
    end
end
