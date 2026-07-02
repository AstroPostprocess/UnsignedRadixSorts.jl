for (KeyT, NPasses) in (
        (UInt8,   1),
        (UInt16,  2),
        (UInt32,  4),
        (UInt64,  8),
    )
    @eval begin
        function UnsignedRadixSorts.onesweep_sort!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {CodeV <: CuVector{$KeyT}, WorkspaceKeyV <: CuVector{$KeyT}, OffsetV <: CuVector{UInt32}, TileSize, NBlocks, ThreadsPerBlock}
            TileSize > 0 || throw(ArgumentError("TileSize must be positive"))
            NBlocks > 0 || throw(ArgumentError("NBlocks must be positive"))
            ThreadsPerBlock >= 32 || throw(ArgumentError("ThreadsPerBlock must be at least one CUDA warp"))
            ThreadsPerBlock % 32 == 0 || throw(ArgumentError("CUDA BlockRadixRank path requires ThreadsPerBlock to be a multiple of 32"))
            ThreadsPerBlock <= 256 || throw(ArgumentError("CUDA BlockRadixRank path currently supports ThreadsPerBlock <= 256"))

            # Work partitioning and dynamic shared-memory size for every pass.
            nelems = length(codes)
            ntiles = cld(nelems, TileSize)
            temp_storage_bytes = _onesweep_pass_shmem_bytes($KeyT, Val(TileSize), Val(ThreadsPerBlock))

            # Allocate/reuse global workspace and prepare the pass-wide bucket
            # starts used by decoupled lookback.
            initialize_base_workspace!(ws, nelems, ntiles)
            prepare_bucket_offsets!(ws, codes, Val(NBlocks), Val(ThreadsPerBlock))

            # Launch one 8-bit OneSweep pass per byte of the key type.
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                CUDA.@cuda threads=ThreadsPerBlock blocks=NBlocks shmem=temp_storage_bytes onesweep_pass_kernel!(
                    codes,
                    ws,
                    Val(TileSize),
                    Val(ThreadsPerBlock),
                    Val(digit),
                )
            end

            # Odd pass counts leave the final keys in the workspace buffer.
            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))
            return nothing
        end

        function UnsignedRadixSorts.onesweep_sort!(codes :: KeyV, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {KeyV <: CuVector{$KeyT}, TileSize, NBlocks, ThreadsPerBlock}
            Workspace = OnesweepWorkspace(CuVector{$KeyT})
            return onesweep_sort!(codes, Workspace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {CodeV <: CuVector{$KeyT}, WorkspaceKeyV <: CuVector{$KeyT}, OffsetV <: CuVector{UInt32}, TileSize, NBlocks, ThreadsPerBlock}
            TileSize > 0 || throw(ArgumentError("TileSize must be positive"))
            NBlocks > 0 || throw(ArgumentError("NBlocks must be positive"))
            ThreadsPerBlock >= 32 || throw(ArgumentError("ThreadsPerBlock must be at least one CUDA warp"))
            ThreadsPerBlock % 32 == 0 || throw(ArgumentError("CUDA BlockRadixRank path requires ThreadsPerBlock to be a multiple of 32"))
            ThreadsPerBlock <= 256 || throw(ArgumentError("CUDA BlockRadixRank path currently supports ThreadsPerBlock <= 256"))

            # Work partitioning and dynamic shared-memory size for every pass.
            nelems = length(codes)
            ntiles = cld(nelems, TileSize)
            temp_storage_bytes = _onesweep_pass_shmem_bytes($KeyT, Val(TileSize), Val(ThreadsPerBlock))

            # Initialize key/value ping-pong workspace and precompute the
            # pass-wide bucket starts shared by all permutation passes.
            initialize_perm_workspace!(ws, nelems, ntiles, Val(NBlocks), Val(ThreadsPerBlock), Val(TileSize))
            prepare_bucket_offsets!(ws, codes, Val(NBlocks), Val(ThreadsPerBlock))

            # Launch one 8-bit OneSweep key/value pass per byte of the key type.
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                CUDA.@cuda threads=ThreadsPerBlock blocks=NBlocks shmem=temp_storage_bytes onesweep_perm_pass_kernel!(
                    codes,
                    ws,
                    Val(TileSize),
                    Val(ThreadsPerBlock),
                    Val(digit),
                )
            end

            # Odd pass counts leave the final keys in the workspace buffer;
            # the returned permutation side must match that final key side.
            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))
            return $(isodd(NPasses) ? :(ws.perms[2]) : :(ws.perms[1]))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: KeyV, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {KeyV <: CuVector{$KeyT}, TileSize, NBlocks, ThreadsPerBlock}
            Workspace = OnesweepWorkspace(CuVector{$KeyT})
            return onesweep_sortperm!(codes, Workspace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
        end
    end
end
