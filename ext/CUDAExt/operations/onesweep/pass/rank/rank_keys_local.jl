using StaticArrays: MVector

## CUDA BlockRadixRank-style local rank helper for OneSweep.
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
## - returned `ranks[item]`  ~= stable tile-local rank

"""
    _rank_keys_early_counts!(keys, warp_offsets, local_counts, local_offsets, lookback, tile_id, tile_len, ::Val{TileSize}, ::Val{ThreadsPerBlock}, ::Val{Pass})

Compute stable tile-local ranks and early per-bucket counts from cached keys.

The block may contain any whole number of CUDA warps up to 256 threads. Threads
cooperatively own the 256 radix buckets in strided order, so configurations such
as 64, 128, or 256 threads use the same algorithm.

CUB parallel: `BlockRadixRankMatchEarlyCounts::RankKeys`. The counts callback
is represented by the early `local_counts` reduction and
`_publish_lookback_partial!` call inside this helper.

# Parameters

- `keys`: Per-thread cached keys in warp-striped input order.
- `warp_offsets`: Shared per-warp bucket counts, later overwritten with cursors.
- `local_counts`: Shared tile counts for all 256 buckets.
- `local_offsets`: Shared exclusive tile prefixes for all 256 buckets.
- `lookback`: Packed global OneSweep lookback table.
- `tile_id`: Current 0-based tile id.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerBlock}`: Compile-time CUDA block size.
- `::Val{Pass}`: Compile-time radix pass selector.

# Returns

- Per-thread stable tile-local ranks corresponding to `keys`.
"""
function _rank_keys_early_counts! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _rank_keys_early_counts!(keys :: MVector{ItemsPerThread, $KeyT}, warp_offsets :: SharedV, local_counts :: SharedV, local_offsets :: SharedV, lookback :: OffsetV, tile_id :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerBlock}, :: Val{Pass}) where {ItemsPerThread, SharedV <: CuDeviceVector{UInt32}, OffsetV <: CuDeviceVector{UInt32}, TileSize, ThreadsPerBlock, Pass}
            NWarps = ThreadsPerBlock ÷ 32

    # Per-thread rank output, aligned with the cached `keys` item slots.
    ranks = MVector{ItemsPerThread, UInt32}(undef)

    # Get this CUDA thread's block and warp coordinates.
    thread_id = Int(CUDA.threadIdx().x)
    nthreads = Int(CUDA.blockDim().x)
    warp_threads = Int(CUDA.warpsize())
    warp_id = fld(thread_id - 1, warp_threads)
    lane_in_warp = (thread_id - 1) % warp_threads
    warp_lane_id = Int(CUDA.laneid())
    full_mask = CUDA.FULL_MASK

    # -----------------------------------------------------
    # 1. RankKeys(): ComputeHistogramsWarp(keys) plus CountsCallback bins.
    #
    # CUB groups lanes with the same digit, then increments a per-warp bucket
    # count once per peer group. `warp_offsets` first stores those per-warp
    # counts; a later stage overwrites the same storage with cursor ranges.
    item = 0
    while item < ItemsPerThread
        # Map this thread/item slot to a tile-local input index.
        local_j = warp_id * warp_threads * ItemsPerThread + item * warp_threads + lane_in_warp + 1
        valid = local_j <= tile_len

        # Extract the current pass digit only for valid tile entries.
        digit = UInt32(0)
        bucket = 1
        if valid
            @inbounds key = keys[item + 1]
            bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
            digit = UInt32(bucket - 1)
        end

        # Group same-digit lanes so the warp issues one atomic per peer group.
        peer_mask = _match_any(full_mask, digit, valid)
        leader_lane = valid ? Int(CUDA.ffs(peer_mask)) : 1
        peer_count = UInt32(CUDA.popc(peer_mask))

        # The leader lane contributes the whole peer group to the warp bucket.
        if valid && warp_lane_id == leader_lane
            idx = warp_id * 256 + bucket
            CUDA.atomic_add!(pointer(warp_offsets, idx), peer_count)
        end

        item += 1
    end
    CUDA.sync_threads()

    # -----------------------------------------------------
    # 2. CountsCallback: reduce per-warp counts into tile bins.
    #
    # `local_counts[bucket]` is the per-tile `bins` payload later published to
    # d_lookback as PARTIAL and then upgraded to GLOBAL by lookback resolution.
    bucket = thread_id
    while bucket <= 256
        # Each thread reduces one or more buckets across all block warps.
        tile_count = zero(UInt32)
        warp = 0
        while warp < NWarps
            idx = warp * 256 + bucket
            @inbounds tile_count += warp_offsets[idx]
            warp += 1
        end
        @inbounds local_counts[bucket] = tile_count
        bucket += nthreads
    end
    CUDA.sync_threads()

    # 3. CountsCallback -> LookbackPartial.
    #
    # CUB publishes PARTIAL as soon as RankKeys knows the bins. Doing it here
    # lets later tiles observe this tile while the current block finishes the
    # rank/scan work.
    _publish_lookback_partial!(lookback, local_counts, tile_id)

    # -----------------------------------------------------
    # 4. RankKeys(): bins -> exclusive_digit_prefix.
    #
    # The radix digit count is fixed at 256, so a single thread can produce the
    # exclusive prefixes cheaply and avoids shared-memory aliasing/scratch races
    # in this correctness-critical path.
    if thread_id == 1
        running = zero(UInt32)
        bucket = 1
        while bucket <= 256
            @inbounds count = local_counts[bucket]
            @inbounds local_offsets[bucket] = running
            running += count
            bucket += 1
        end
    end
    CUDA.sync_threads()

    # -----------------------------------------------------
    # 5. RankKeys(): ComputeOffsetsWarpDownsweep(exclusive_digit_prefix).
    #
    # Each bucket starts from `local_offsets[bucket]`. Warps receive disjoint
    # cursor ranges for that bucket in warp order, preserving tile stability.
    bucket = thread_id
    while bucket <= 256
        # Replace each warp's count with that warp's starting cursor.
        @inbounds running = local_offsets[bucket]
        warp = 0
        while warp < NWarps
            idx = warp * 256 + bucket
            @inbounds count = warp_offsets[idx]
            @inbounds warp_offsets[idx] = running
            running += count
            warp += 1
        end
        bucket += nthreads
    end
    CUDA.sync_threads()

    # -----------------------------------------------------
    # 6. RankKeys(): ComputeRanksItem(keys, ranks, WARP_MATCH_ANY).
    #
    # Warp order, item order, and lane peer prefixes match the warp-striped
    # input order used by `_load_keys!`. The peer-group leader advances the
    # warp bucket cursor once, then every peer adds its lane-local prefix.
    lane_mask_lt = CUDA.lanemask(<)
    item = 0
    while item < ItemsPerThread
        # Recompute the same tile-local input index and digit for rank output.
        local_j = warp_id * warp_threads * ItemsPerThread + item * warp_threads + lane_in_warp + 1
        valid = local_j <= tile_len

        digit = UInt32(0)
        bucket_for_key = 1
        if valid
            @inbounds key = keys[item + 1]
            bucket_for_key = UnsignedRadixSorts._radix_bucket(key, Pass)
            digit = UInt32(bucket_for_key - 1)
        end

        # Peer prefix is the number of same-digit lanes before this lane.
        peer_mask = _match_any(full_mask, digit, valid)
        leader_lane = valid ? Int(CUDA.ffs(peer_mask)) : 1
        peer_count = UInt32(CUDA.popc(peer_mask))
        peer_prefix = UInt32(CUDA.popc(peer_mask & lane_mask_lt))
        warp_bucket_prefix = zero(UInt32)

        if valid && warp_lane_id == leader_lane
            idx = warp_id * 256 + bucket_for_key
            warp_bucket_prefix = CUDA.atomic_add!(pointer(warp_offsets, idx), peer_count)
        end

        # Broadcast the leader's cursor to the whole peer group.
        warp_bucket_prefix = CUDA.shfl_sync(full_mask, warp_bucket_prefix, leader_lane)
        rank = valid ? warp_bucket_prefix + peer_prefix : zero(UInt32)
        @inbounds ranks[item + 1] = rank
        item += 1
    end

    return ranks
end
    end
end
