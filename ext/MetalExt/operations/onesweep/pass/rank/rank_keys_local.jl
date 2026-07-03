using StaticArrays: MVector

## Metal BlockRadixRank-style local rank helper for OneSweep.
##
## This file follows the CUDA implementation stage-for-stage. The only backend
## substitution is `_match_any`/peer broadcast, which currently use SIMD-local
## threadgroup scratch because Metal.jl does not expose `simd_ballot` or an
## arbitrary-source SIMD shuffle.

"""
    _exclusive_scan(local_counts, local_offsets, scan_scratch, ::Val{ThreadsPerGroup})

Compute the exclusive prefix sum of the 256 radix-bin counts in parallel.

Each thread first scans a contiguous subset of radix bins. SIMD-group scans
then combine the per-thread totals, and the first SIMD group scans the group
totals to produce threadgroup-wide prefixes.

# Parameters

- `local_counts`: Threadgroup-memory counts for the 256 radix buckets.
- `local_offsets`: Threadgroup-memory exclusive prefixes written in-place.
- `scan_scratch`: Threadgroup scratch with at least one `UInt32` per SIMD group.
- `::Val{ThreadsPerGroup}`: Compile-time Metal threadgroup size.
"""
@inline function _exclusive_scan(local_counts :: CountsV, local_offsets :: OffsetsV, scan_scratch :: ScratchV, :: Val{ThreadsPerGroup}) where {CountsV <: MtlDeviceVector{UInt32}, OffsetsV <: MtlDeviceVector{UInt32}, ScratchV <: MtlDeviceVector{UInt32}, ThreadsPerGroup}
    BinsPerThread = cld(256, ThreadsPerGroup)
    NSimdgroups = ThreadsPerGroup ÷ 32

    thread_id = Int(Metal.thread_position_in_threadgroup().x)
    simd_threads = Int(Metal.threads_per_simdgroup())
    simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
    lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

    thread_total = zero(UInt32)
    bin_item = 0
    while bin_item < BinsPerThread
        bucket = (thread_id - 1) * BinsPerThread + bin_item + 1
        if bucket <= 256
            @inbounds count = local_counts[bucket]
            @inbounds local_offsets[bucket] = thread_total
            thread_total += count
        end
        bin_item += 1
    end

    inclusive = thread_total
    offset = 1
    while offset < simd_threads
        addend = Metal.simd_shuffle_up(inclusive, offset)
        if lane_in_simd >= offset
            inclusive += addend
        end
        offset <<= 1
    end

    if lane_in_simd == simd_threads - 1
        @inbounds scan_scratch[simd_id + 1] = inclusive
    end
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    if simd_id == 0
        simd_total = zero(UInt32)
        if lane_in_simd < NSimdgroups
            @inbounds simd_total = scan_scratch[lane_in_simd + 1]
        end

        simd_inclusive = simd_total
        offset = 1
        while offset < simd_threads
            addend = Metal.simd_shuffle_up(simd_inclusive, offset)
            if lane_in_simd >= offset
                simd_inclusive += addend
            end
            offset <<= 1
        end

        if lane_in_simd < NSimdgroups
            @inbounds scan_scratch[lane_in_simd + 1] = simd_inclusive - simd_total
        end
    end
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    @inbounds simd_prefix = scan_scratch[simd_id + 1]
    thread_prefix = simd_prefix + inclusive - thread_total

    bin_item = 0
    while bin_item < BinsPerThread
        bucket = (thread_id - 1) * BinsPerThread + bin_item + 1
        if bucket <= 256
            @inbounds local_offsets[bucket] += thread_prefix
        end
        bin_item += 1
    end

    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    return nothing
end

"""
    _rank_keys_early_counts!(ranks, keys, simd_offsets, local_counts, local_offsets, scan_scratch, match_scratch, lookback, tile_id, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup}, ::Val{Pass})

Compute stable tile-local ranks and early per-bucket counts from cached keys.

The stages and data ownership follow the CUDA implementation of
`BlockRadixRankMatchEarlyCounts::RankKeys`: peer-group histogram, early counts
publication, exclusive digit scan, per-SIMD cursor downsweep, then stable rank
assignment.

# Returns

"""
function _rank_keys_early_counts! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _rank_keys_early_counts!(ranks :: MVector{ItemsPerThread, UInt32}, keys :: MVector{ItemsPerThread, $KeyT}, simd_offsets :: SharedV, local_counts :: SharedV, local_offsets :: SharedV, scan_scratch :: SharedV, match_scratch :: SharedV, lookback :: OffsetV, tile_id :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {ItemsPerThread, SharedV <: MtlDeviceVector{UInt32}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            NSimdgroups = ThreadsPerGroup ÷ 32

            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

            # 1. ComputeHistogramsWarp/Simdgroup plus CountsCallback bins.
            item = 0
            while item < ItemsPerThread
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                valid = local_j <= tile_len

                digit = UInt32(0)
                bucket = 1
                if valid
                    @inbounds key = keys[item + 1]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    digit = UInt32(bucket - 1)
                end

                peer_mask = _match_any(digit, valid, match_scratch, simd_id, lane_in_simd, simd_threads)
                leader_lane = valid ? trailing_zeros(peer_mask) : lane_in_simd
                peer_count = UInt32(count_ones(peer_mask))

                if valid && lane_in_simd == leader_lane
                    idx = simd_id * 256 + bucket
                    Metal.atomic_fetch_add_explicit(pointer(simd_offsets, idx), peer_count)
                end

                item += 1
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # 2. Reduce per-SIMD counts into tile bins.
            bucket = thread_id
            while bucket <= 256
                tile_count = zero(UInt32)
                simd = 0
                while simd < NSimdgroups
                    idx = simd * 256 + bucket
                    @inbounds tile_count += simd_offsets[idx]
                    simd += 1
                end
                @inbounds local_counts[bucket] = tile_count
                bucket += nthreads
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # 3. CountsCallback -> LookbackPartial.
            _publish_lookback_partial!(lookback, local_counts, tile_id)

            # 4. bins -> exclusive_digit_prefix.
            _exclusive_scan(local_counts, local_offsets, scan_scratch, Val(ThreadsPerGroup))

            # 5. ComputeOffsetsWarpDownsweep.
            bucket = thread_id
            while bucket <= 256
                @inbounds running = local_offsets[bucket]
                simd = 0
                while simd < NSimdgroups
                    idx = simd * 256 + bucket
                    @inbounds count = simd_offsets[idx]
                    @inbounds simd_offsets[idx] = running
                    running += count
                    simd += 1
                end
                bucket += nthreads
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # 6. ComputeRanksItem with stable SIMD/item/lane ordering.
            lane_mask_lt = lane_in_simd == 0 ? zero(UInt32) : (UInt32(1) << UInt32(lane_in_simd)) - UInt32(1)
            item = 0
            while item < ItemsPerThread
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                valid = local_j <= tile_len

                digit = UInt32(0)
                bucket_for_key = 1
                if valid
                    @inbounds key = keys[item + 1]
                    bucket_for_key = UnsignedRadixSorts._radix_bucket(key, Pass)
                    digit = UInt32(bucket_for_key - 1)
                end

                peer_mask = _match_any(digit, valid, match_scratch, simd_id, lane_in_simd, simd_threads)
                leader_lane = valid ? trailing_zeros(peer_mask) : lane_in_simd
                peer_count = UInt32(count_ones(peer_mask))
                peer_prefix = UInt32(count_ones(peer_mask & lane_mask_lt))
                simd_bucket_prefix = zero(UInt32)

                if valid && lane_in_simd == leader_lane
                    idx = simd_id * 256 + bucket_for_key
                    simd_bucket_prefix = Metal.atomic_fetch_add_explicit(pointer(simd_offsets, idx), peer_count)
                end

                simd_bucket_prefix = _broadcast_peer_leader(simd_bucket_prefix, leader_lane, match_scratch, simd_id, lane_in_simd, simd_threads)
                @inbounds ranks[item + 1] = valid ? simd_bucket_prefix + peer_prefix : zero(UInt32)
                item += 1
            end

            return nothing
        end
    end
end
