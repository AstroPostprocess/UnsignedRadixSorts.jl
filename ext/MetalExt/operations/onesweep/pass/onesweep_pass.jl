using StaticArrays: MVector

## Metal port of CCCL's AgentRadixSortOnesweep::Process().
##
## The stage order, per-thread cached keys/ranks, early counts publication,
## shared key staging, lookback, and global scatter follow the CUDA version.
## Metal-specific substitutions are limited to backend intrinsics. The default
## Metal policy uses TileSize=2048 to fit the 32 KiB threadgroup-memory budget.

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function onesweep_pass_kernel!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            NSimdgroups = ThreadsPerGroup ÷ 32
            ItemsPerThread = cld(TileSize, ThreadsPerGroup)

            src, dst = _select_pass_key_buffers(codes, ws, Val(Pass))
            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            simd_offsets = Metal.MtlThreadGroupArray(UInt32, NSimdgroups * 256)
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            scan_scratch = Metal.MtlThreadGroupArray(UInt32, NSimdgroups)
            match_scratch = Metal.MtlThreadGroupArray(UInt32, ThreadsPerGroup)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            keys_out = Metal.MtlThreadGroupArray($KeyT, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)

            # Threadgroup memory with UInt64 keys and ThreadsPerGroup=256 is
            # 28,708 bytes, below Metal's 32 KiB limit.
            while true
                tile_id = _claim_next_tile!(ws.tile_counter, claimed_tile)
                tile_id < ntiles || break

                rangemin = tile_id * TileSize + 1
                tile_len = min(TileSize, nelems - rangemin + 1)

                _clear_rank_storage!(simd_offsets)
                keys = zero(MVector{ItemsPerThread, $KeyT})
                ranks = zero(MVector{ItemsPerThread, UInt32})
                _load_keys!(keys, src, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup))
                _rank_keys_early_counts!(ranks, keys, simd_offsets, local_counts, local_offsets, scan_scratch, match_scratch, ws.lookback, tile_id, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))

                _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize), Val(ThreadsPerGroup))
                _resolve_lookback_global_offsets!(ws.lookback, ws.bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, Val(Pass))
                _scatter_keys_global!(keys_out, dst, global_offsets, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))
            end

            return nothing
        end

        @inline function onesweep_perm_pass_kernel!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize}, :: Val{ThreadsPerGroup}, :: Val{Pass}) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, TileSize, ThreadsPerGroup, Pass}
            NSimdgroups = ThreadsPerGroup ÷ 32
            ItemsPerThread = cld(TileSize, ThreadsPerGroup)

            src, dst, perm_src, perm_dst = _select_pass_key_value_buffers(codes, ws, Val(Pass))
            nelems = length(src)
            ntiles = cld(nelems, TileSize)

            simd_offsets = Metal.MtlThreadGroupArray(UInt32, NSimdgroups * 256)
            local_counts = Metal.MtlThreadGroupArray(UInt32, 256)
            local_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            scan_scratch = Metal.MtlThreadGroupArray(UInt32, NSimdgroups)
            match_scratch = Metal.MtlThreadGroupArray(UInt32, ThreadsPerGroup)
            global_offsets = Metal.MtlThreadGroupArray(UInt32, 256)
            keys_out = Metal.MtlThreadGroupArray($KeyT, TileSize)
            claimed_tile = Metal.MtlThreadGroupArray(UInt32, 1)
            $(sizeof(KeyT) < sizeof(UInt32) ? :(values_out = Metal.MtlThreadGroupArray(UInt32, TileSize)) : :(nothing))

            # The UInt64 sortperm path has the same 28,708-byte threadgroup
            # footprint as key-only sorting because it reuses per-thread keys
            # as the value staging source.
            while true
                tile_id = _claim_next_tile!(ws.tile_counter, claimed_tile)
                tile_id < ntiles || break

                rangemin = tile_id * TileSize + 1
                tile_len = min(TileSize, nelems - rangemin + 1)

                _clear_rank_storage!(simd_offsets)
                keys = zero(MVector{ItemsPerThread, $KeyT})
                ranks = zero(MVector{ItemsPerThread, UInt32})
                _load_keys!(keys, src, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup))
                _rank_keys_early_counts!(ranks, keys, simd_offsets, local_counts, local_offsets, scan_scratch, match_scratch, ws.lookback, tile_id, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))

                _scatter_keys_shared!(keys_out, keys, ranks, tile_len, Val(TileSize), Val(ThreadsPerGroup))
                _resolve_lookback_global_offsets!(ws.lookback, ws.bucket_offsets, local_counts, local_offsets, global_offsets, tile_id, Val(Pass))
                $(sizeof(KeyT) < sizeof(UInt32) ?
                    :(_scatter_key_values_global!(keys_out, values_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))) :
                    :(_scatter_key_values_global!(keys_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, Val(TileSize), Val(ThreadsPerGroup), Val(Pass))))
            end

            return nothing
        end
    end
end
