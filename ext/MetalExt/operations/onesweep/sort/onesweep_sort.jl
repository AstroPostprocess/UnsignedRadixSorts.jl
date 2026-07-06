for (KeyT, NPasses) in (
        (UInt8,   1),
        (UInt16,  2),
        (UInt32,  4),
        (UInt64,  8),
    )
    @eval begin
        function UnsignedRadixSorts.onesweep_sort!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, WorkspaceKeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, TileSize, NThreadgroups, ThreadsPerGroup}
            TileSize > 0 || throw(ArgumentError("TileSize must be positive"))
            NThreadgroups > 0 || throw(ArgumentError("NThreadgroups must be positive"))
            ThreadsPerGroup >= 32 || throw(ArgumentError("ThreadsPerGroup must be at least one Metal SIMD group"))
            ThreadsPerGroup % 32 == 0 || throw(ArgumentError("Metal BlockRadixRank path requires ThreadsPerGroup to be a multiple of 32"))
            ThreadsPerGroup <= 256 || throw(ArgumentError("Metal BlockRadixRank path currently supports ThreadsPerGroup <= 256"))

            # Work partitioning for every pass.
            nelems = length(codes)
            nelems == 0 && return nothing
            ntiles = cld(nelems, TileSize)

            # Allocate/reuse global workspace and prepare the pass-wide bucket
            # starts used by decoupled lookback.
            resize_base_workspace!(ws, nelems, ntiles)
            prepare_bucket_offsets!(ws, codes, Val(NThreadgroups), Val(ThreadsPerGroup))

            # Launch one 8-bit OneSweep pass per byte of the key type.
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) onesweep_pass_kernel!(codes, ws, Val(TileSize), Val(ThreadsPerGroup), Val(digit))
            end

            # Odd pass counts leave the final keys in the workspace buffer.
            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))
            return nothing
        end

        function UnsignedRadixSorts.onesweep_sort!(codes :: CodeV, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, TileSize, NThreadgroups, ThreadsPerGroup}
            Workspace = OnesweepWorkspace(MtlVector{$KeyT})
            return onesweep_sort!(codes, Workspace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, WorkspaceKeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, TileSize, NThreadgroups, ThreadsPerGroup}
            TileSize > 0 || throw(ArgumentError("TileSize must be positive"))
            NThreadgroups > 0 || throw(ArgumentError("NThreadgroups must be positive"))
            ThreadsPerGroup >= 32 || throw(ArgumentError("ThreadsPerGroup must be at least one Metal SIMD group"))
            ThreadsPerGroup % 32 == 0 || throw(ArgumentError("Metal BlockRadixRank path requires ThreadsPerGroup to be a multiple of 32"))
            ThreadsPerGroup <= 256 || throw(ArgumentError("Metal BlockRadixRank path currently supports ThreadsPerGroup <= 256"))

            # Work partitioning for every pass.
            nelems = length(codes)
            nelems == 0 && return similar(codes, UInt32, 0)
            ntiles = cld(nelems, TileSize)

            # Initialize key/value ping-pong workspace and precompute the
            # pass-wide bucket starts shared by all permutation passes.
            initialize_perm_workspace_for_sort!(ws, nelems, ntiles, Val(NThreadgroups), Val(ThreadsPerGroup))
            prepare_bucket_offsets!(ws, codes, Val(NThreadgroups), Val(ThreadsPerGroup))

            # Launch one 8-bit OneSweep key/value pass per byte of the key type.
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) onesweep_perm_pass_kernel!(codes, ws, Val(TileSize), Val(ThreadsPerGroup), Val(digit))
            end

            # Odd pass counts leave the final keys in the workspace buffer;
            # the returned permutation side must match that final key side.
            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))
            return $(isodd(NPasses) ? :(ws.perms[2]) : :(ws.perms[1]))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: CodeV, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, TileSize, NThreadgroups, ThreadsPerGroup}
            Workspace = OnesweepWorkspace(MtlVector{$KeyT})
            return onesweep_sortperm!(codes, Workspace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
        end
    end
end
