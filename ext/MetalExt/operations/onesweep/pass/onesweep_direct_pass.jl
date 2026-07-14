using StaticArrays: MVector

## Forward-progress-safe Metal radix pass.
##
## CUDA OneSweep fuses rank, decoupled lookback, and scatter in one kernel.
## Metal does not guarantee forward progress between oversubscribed
## threadgroups, so an in-kernel lookback spin can starve its producer.  This
## backend keeps the same rank/count representation but places an implicit
## command-queue boundary between three kernels:
##
##   1. rank every tile, publish PARTIAL counts, persist within-bucket ranks;
##   2. convert PARTIAL counts to GLOBAL inclusive prefixes with one thread per
##      radix bucket;
##   3. scatter every tile without cross-threadgroup communication.
##
## No tile is ranked twice and no kernel waits on another threadgroup.

@inline function _store_within_bucket_ranks!(
        stored_ranks,
        src,
        ranks,
        local_offsets,
        rangemin::Int,
        tile_len::Int,
        ::Val{Pass},
    ) where {Pass}
    # Rank production is SIMD-striped, while persistence below is linearly
    # striped over the threadgroup. Ensure every producer has finished before
    # another lane reads its rank slot.
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    thread_id = Int(Metal.thread_position_in_threadgroup().x)
    nthreads = Int(Metal.threads_per_threadgroup().x)
    local_i = thread_id
    while local_i <= tile_len
        i = rangemin + local_i - 1
        @inbounds begin
            key = src[i]
            bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
            stored_ranks[i] = ranks[local_i] - local_offsets[bucket]
        end
        local_i += nthreads
    end

    # The next claimed tile reuses ranks and local_offsets.
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    return nothing
end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function onesweep_direct_rank_kernel!(
                codes::CodeV,
                ws::OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV},
                ::Val{TileSize},
                ::Val{ThreadsPerGroup},
                ::Val{Pass},
            ) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            NSimdgroups = ThreadsPerGroup ÷ 32
            ItemsPerThread = cld(TileSize, ThreadsPerGroup)
            src, _ = _select_pass_key_buffers(codes, ws, Val(Pass))
            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            simd_offsets = Metal.MtlThreadGroupArray(UInt32, NSimdgroups * 256)
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            scan_scratch = Metal.MtlThreadGroupArray(UInt32, NSimdgroups)
            # AIR ballot/shuffle do not use this compatibility argument.
            match_scratch = Metal.MtlThreadGroupArray(UInt32, 1)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)
            ranks = Metal.MtlThreadGroupArray(UInt32, TileSize)

            while true
                tile_id = _claim_next_tile!(ws.tile_counter, claimed_tile)
                tile_id < ntiles || break
                rangemin = tile_id * TileSize + 1
                tile_len = min(TileSize, nelems - rangemin + 1)

                _clear_rank_storage!(simd_offsets)
                keys = zero(MVector{ItemsPerThread, $KeyT})
                _load_keys!(
                    keys, src, rangemin, tile_len,
                    Val(TileSize), Val(ThreadsPerGroup),
                )
                _rank_keys_early_counts!(
                    ranks,
                    keys,
                    simd_offsets,
                    local_counts,
                    local_offsets,
                    scan_scratch,
                    match_scratch,
                    ws.lookback,
                    tile_id,
                    tile_len,
                    Val(TileSize),
                    Val(ThreadsPerGroup),
                    Val(Pass),
                )
                _store_within_bucket_ranks!(
                    ws.local_ranks,
                    src,
                    ranks,
                    local_offsets,
                    rangemin,
                    tile_len,
                    Val(Pass),
                )
            end
            return nothing
        end

        @inline function onesweep_direct_scatter_kernel!(
                codes::CodeV,
                ws::OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV},
                ::Val{TileSize},
                ::Val{Pass},
            ) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, Pass}
            src, dst = _select_pass_key_buffers(codes, ws, Val(Pass))
            nelems = length(src)
            tile_id = Int(Metal.threadgroup_position_in_grid().x) - 1
            rangemin = tile_id * TileSize + 1
            tile_len = min(TileSize, nelems - rangemin + 1)
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)

            local_i = thread_id
            while local_i <= tile_len
                i = rangemin + local_i - 1
                @inbounds begin
                    key = src[i]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    previous = if tile_id == 0
                        UInt32(0)
                    else
                        idx = UnsignedRadixSorts._lookback_index(tile_id - 1, bucket)
                        UnsignedRadixSorts._entry_count(ws.lookback[idx])
                    end
                    bucket_start = ws.bucket_offsets[
                        UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)
                    ]
                    output_idx = bucket_start + previous + ws.local_ranks[i]
                    dst[Int(output_idx)] = key
                end
                local_i += nthreads
            end
            return nothing
        end

        @inline function onesweep_direct_perm_scatter_kernel!(
                codes::CodeV,
                ws::OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV},
                ::Val{TileSize},
                ::Val{Pass},
            ) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, Pass}
            src, dst, perm_src, perm_dst = _select_pass_key_value_buffers(codes, ws, Val(Pass))
            nelems = length(src)
            tile_id = Int(Metal.threadgroup_position_in_grid().x) - 1
            rangemin = tile_id * TileSize + 1
            tile_len = min(TileSize, nelems - rangemin + 1)
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)

            local_i = thread_id
            while local_i <= tile_len
                i = rangemin + local_i - 1
                @inbounds begin
                    key = src[i]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    previous = if tile_id == 0
                        UInt32(0)
                    else
                        idx = UnsignedRadixSorts._lookback_index(tile_id - 1, bucket)
                        UnsignedRadixSorts._entry_count(ws.lookback[idx])
                    end
                    bucket_start = ws.bucket_offsets[
                        UnsignedRadixSorts._bucket_offsets_index(Pass, bucket)
                    ]
                    output_idx = bucket_start + previous + ws.local_ranks[i]
                    dst[Int(output_idx)] = key
                    perm_dst[Int(output_idx)] = perm_src[i]
                end
                local_i += nthreads
            end
            return nothing
        end
    end
end

@inline function onesweep_direct_prefix_kernel!(lookback, ntiles::Int)
    bucket = Int(Metal.thread_position_in_threadgroup().x)
    running = UInt32(0)
    tile_id = 0
    while tile_id < ntiles
        idx = UnsignedRadixSorts._lookback_index(tile_id, bucket)
        @inbounds running += UnsignedRadixSorts._entry_count(lookback[idx])
        @inbounds lookback[idx] = UnsignedRadixSorts._global_entry(running)
        tile_id += 1
    end
    return nothing
end
