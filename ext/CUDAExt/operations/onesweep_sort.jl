for (KeyT, NPasses) in (
        (UInt8,   1),
        (UInt16,  2),
        (UInt32,  4),
        (UInt64,  8)
    )
    @eval begin
        function UnsignedRadixSorts.onesweep_sort!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {KeyV <: CuVector{$KeyT}, OffsetV <: CuVector{UInt32}, TileSize, NBlocks, ThreadsPerBlock}
            ThreadsPerBlock % 32 == 0 || throw(ArgumentError("CUDA BlockRadixRank path requires ThreadsPerBlock to be a multiple of 32"))
            ThreadsPerBlock <= 256 || throw(ArgumentError("CUDA BlockRadixRank path currently supports ThreadsPerBlock <= 256"))

            # Nelems: number of elements that need to be sorted
            nelems = length(codes)
            # Number of data tiles
            ntiles = cld(nelems, TileSize)
            # Number of CUDA blocks and threads per block are controlled by
            # NBlocks and ThreadsPerBlock.

            # Initialize only the base workspace; local pass scratch lives in
            # CUDA shared memory.
            initialize_base_workspace!(ws, nelems, ntiles)

            # Before the first pass: global histogram for bucket
            prepare_bucket_offsets!(ws, codes, Val(NBlocks), Val(ThreadsPerBlock))

            # Onesweep passes, 8 bits per pass for UInt64
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                CUDA.@cuda threads=ThreadsPerBlock blocks=NBlocks onesweep_pass_kernel!(codes, ws, Val(TileSize), Val(digit))
            end
            CUDA.synchronize()

            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))

            return nothing
        end

        function UnsignedRadixSorts.onesweep_sort!(codes :: KeyV, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {KeyV <: CuVector{$KeyT}, TileSize, NBlocks, ThreadsPerBlock}
            Workspace = OnesweepWorkspace(CuVector{$KeyT})
            return onesweep_sort!(codes, Workspace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {KeyV <: CuVector{$KeyT}, OffsetV <: CuVector{UInt32}, TileSize, NBlocks, ThreadsPerBlock}
            ThreadsPerBlock % 32 == 0 || throw(ArgumentError("CUDA BlockRadixRank path requires ThreadsPerBlock to be a multiple of 32"))
            ThreadsPerBlock <= 256 || throw(ArgumentError("CUDA BlockRadixRank path currently supports ThreadsPerBlock <= 256"))

            # Nelems: number of elements that need to be sorted
            nelems = length(codes)
            # Number of data tiles
            ntiles = cld(nelems, TileSize)
            # Number of CUDA blocks and threads per block are controlled by
            # NBlocks and ThreadsPerBlock.

            # Initialize workspace
            initialize_perm_workspace!(ws, nelems, ntiles, Val(NBlocks), Val(ThreadsPerBlock), Val(TileSize))

            # Before the first pass: global histogram for bucket
            prepare_bucket_offsets!(ws, codes, Val(NBlocks), Val(ThreadsPerBlock))

            # Onesweep passes, 8 bits per pass for UInt64
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                CUDA.@cuda threads=ThreadsPerBlock blocks=NBlocks onesweep_perm_pass_kernel!(codes, ws, Val(TileSize), Val(digit))
            end
            CUDA.synchronize()

            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))

            return $(isodd(NPasses) ? :(ws.perms[2]) : :(ws.perms[1]))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: KeyV, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {KeyV <: CuVector{$KeyT}, TileSize, NBlocks, ThreadsPerBlock}
            Workspace = OnesweepWorkspace(CuVector{$KeyT})
            return onesweep_sortperm!(codes, Workspace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
        end

    end
end
