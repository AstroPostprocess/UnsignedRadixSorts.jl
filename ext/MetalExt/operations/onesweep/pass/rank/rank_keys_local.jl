using StaticArrays: MVector

## Metal BlockRadixRank-style local rank helper for OneSweep.
##
##   Upstream CCCL: cub/cub/block/block_radix_rank.cuh
##   cub::BlockRadixRankMatchEarlyCounts::RankKeys
##
## This helper implements the OneSweep pass step:
##
##   BlockRadixRankT(s.rank_temp_storage)
##       .RankKeys(keys, ranks, digit_extractor(),
##                 exclusive_digit_prefix,
##                 CountsCallback(*this, bins, keys));
##
## Outputs:
## - `local_counts[bucket]`  ~= CUB `bins`
## - `local_offsets[bucket]` ~= CUB `exclusive_digit_prefix`
## - `ranks[item]`           ~= stable tile-local rank
##
## The only backend substitution from CUDA is `_match_any`/peer broadcast,
## which currently use SIMD-local threadgroup scratch because Metal.jl does not
## expose `simd_ballot` or an arbitrary-source SIMD shuffle.

"""
    _exclusive_scan(local_counts, local_offsets, scan_scratch, ::Val{ThreadsPerGroup})

Compute the exclusive prefix sum of the 256 radix-bin counts in parallel.

Each thread first scans a contiguous subset of radix bins. SIMD-group scans
then combine the per-thread totals, and the first SIMD group scans the group
totals to produce threadgroup-wide prefixes. During this step `scan_scratch` is
only temporary scan storage; lookback fills `global_offsets` with true global
scatter bases after RankKeys completes.

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

    # Scan the contiguous bucket range owned by this thread.
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

    # Inclusive scan of per-thread totals within each SIMD group.
    inclusive = thread_total
    offset = 1
    while offset < simd_threads
        addend = Metal.simd_shuffle_up(inclusive, offset)
        if lane_in_simd >= offset
            inclusive += addend
        end
        offset <<= 1
    end

    # Store one total per SIMD group.
    if lane_in_simd == simd_threads - 1
        @inbounds scan_scratch[simd_id + 1] = inclusive
    end
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    # The first SIMD group computes exclusive prefixes of the SIMD totals.
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

    # Add the threadgroup-wide prefix of this thread to its local bucket prefixes.
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

The threadgroup may contain any whole number of Metal SIMD groups up to 256
threads. Threads cooperatively own the 256 radix buckets in strided order, so
configurations such as 64, 128, or 256 threads use the same algorithm.

CUB parallel: `BlockRadixRankMatchEarlyCounts::RankKeys`. The counts callback
is represented by the early `local_counts` reduction and
`_publish_lookback_partial!` call inside this helper.

# Parameters

- `ranks`: Per-thread stable tile-local ranks corresponding to `keys`.
- `keys`: Per-thread cached keys in SIMD-striped input order.
- `simd_offsets`: Threadgroup per-SIMD bucket counts, later overwritten with cursors.
- `local_counts`: Threadgroup tile counts for all 256 buckets.
- `local_offsets`: Threadgroup exclusive tile prefixes for all 256 buckets.
- `scan_scratch`: Threadgroup scan storage used by `_exclusive_scan`.
- `match_scratch`: Threadgroup scratch used by Metal `_match_any` and peer broadcast.
- `lookback`: Packed global OneSweep lookback table.
- `tile_id`: Current 0-based tile id.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerGroup}`: Compile-time Metal threadgroup size.
- `::Val{Pass}`: Compile-time radix pass selector.

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

            # -----------------------------------------------------
            # 1. RankKeys(): ComputeHistogramsWarp(keys) plus CountsCallback bins.
            #
            # CUB groups lanes with the same digit, then increments a per-warp bucket
            # count once per peer group. `simd_offsets` first stores those per-SIMD
            # counts; a later stage overwrites the same storage with cursor ranges.
            item = 0
            while item < ItemsPerThread
                # Map this thread/item slot to a tile-local input index.
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                valid = local_j <= tile_len

                # Extract the current pass digit only for valid tile entries.
                digit = UInt32(0)
                bucket = 1
                if valid
                    @inbounds key = keys[item + 1]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    digit = UInt32(bucket - 1)
                end

                # Group same-digit lanes so the SIMD group issues one atomic per peer group.
                peer_mask = _match_any(digit, valid, match_scratch, simd_id, lane_in_simd, simd_threads)
                leader_lane = valid ? trailing_zeros(peer_mask) : lane_in_simd
                peer_count = UInt32(count_ones(peer_mask))

                # The leader lane contributes the whole peer group to the SIMD bucket.
                if valid && lane_in_simd == leader_lane
                    idx = simd_id * 256 + bucket
                    Metal.atomic_fetch_add_explicit(pointer(simd_offsets, idx), peer_count)
                end

                item += 1
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # -----------------------------------------------------
            # 2. CountsCallback: reduce per-SIMD counts into tile bins.
            #
            # `local_counts[bucket]` is the per-tile `bins` payload later published to
            # d_lookback as PARTIAL and then upgraded to GLOBAL by lookback resolution.
            bucket = thread_id
            while bucket <= 256
                # Each thread reduces one or more buckets across all SIMD groups.
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
            #
            # CUB publishes PARTIAL as soon as RankKeys knows the bins. Doing it here
            # lets later tiles observe this tile while the current threadgroup finishes
            # the rank/scan work.
            _publish_lookback_partial!(lookback, local_counts, tile_id)

            # -----------------------------------------------------
            # 4. RankKeys(): bins -> exclusive_digit_prefix.
            #
            # Threads scan contiguous bucket ranges, then a hierarchical
            # SIMD/threadgroup scan converts those local prefixes into
            # threadgroup-wide exclusive offsets.
            _exclusive_scan(local_counts, local_offsets, scan_scratch, Val(ThreadsPerGroup))

            # -----------------------------------------------------
            # 5. RankKeys(): ComputeOffsetsWarpDownsweep(exclusive_digit_prefix).
            #
            # Each bucket starts from `local_offsets[bucket]`. SIMD groups receive
            # disjoint cursor ranges for that bucket in SIMD-group order, preserving
            # tile stability.
            bucket = thread_id
            while bucket <= 256
                # Replace each SIMD group's count with that SIMD group's starting cursor.
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

            # -----------------------------------------------------
            # 6. RankKeys(): ComputeRanksItem(keys, ranks, WARP_MATCH_ANY).
            #
            # SIMD-group order, item order, and lane peer prefixes match the
            # SIMD-striped input order used by `_load_keys!`. The peer-group leader
            # advances the SIMD bucket cursor once, then every peer adds its
            # lane-local prefix.
            lane_mask_lt = lane_in_simd == 0 ? zero(UInt32) : (UInt32(1) << UInt32(lane_in_simd)) - UInt32(1)
            item = 0
            while item < ItemsPerThread
                # Recompute the same tile-local input index and digit for rank output.
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                valid = local_j <= tile_len

                digit = UInt32(0)
                bucket_for_key = 1
                if valid
                    @inbounds key = keys[item + 1]
                    bucket_for_key = UnsignedRadixSorts._radix_bucket(key, Pass)
                    digit = UInt32(bucket_for_key - 1)
                end

                # Peer prefix is the number of same-digit lanes before this lane.
                peer_mask = _match_any(digit, valid, match_scratch, simd_id, lane_in_simd, simd_threads)
                leader_lane = valid ? trailing_zeros(peer_mask) : lane_in_simd
                peer_count = UInt32(count_ones(peer_mask))
                peer_prefix = UInt32(count_ones(peer_mask & lane_mask_lt))
                simd_bucket_prefix = zero(UInt32)

                if valid && lane_in_simd == leader_lane
                    idx = simd_id * 256 + bucket_for_key
                    simd_bucket_prefix = Metal.atomic_fetch_add_explicit(pointer(simd_offsets, idx), peer_count)
                end

                # Broadcast the leader's cursor to the whole peer group.
                simd_bucket_prefix = _broadcast_peer_leader(simd_bucket_prefix, leader_lane, match_scratch, simd_id, lane_in_simd, simd_threads)
                @inbounds ranks[item + 1] = valid ? simd_bucket_prefix + peer_prefix : zero(UInt32)
                item += 1
            end

            return nothing
        end
    end
end
