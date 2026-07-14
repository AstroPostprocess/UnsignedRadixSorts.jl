for (KeyT, NPasses) in (
        (UInt8,   1),
        (UInt16,  2),
        (UInt32,  4),
        (UInt64,  8),
    )
    @eval begin
        function UnsignedRadixSorts.radix_sort!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, WorkspaceKeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, TileSize, NThreadgroups, ThreadsPerGroup}
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
            # starts shared by the direct-prefix passes.
            resize_base_workspace!(ws, nelems, ntiles)
            prepare_bucket_offsets!(ws, codes, Val(NThreadgroups), Val(ThreadsPerGroup))

            # Launch one forward-progress-safe 8-bit pass per byte. Rank/count,
            # bucket-prefix finalization, and scatter are separate dispatches;
            # no Metal threadgroup waits for another threadgroup in-kernel.
            rank_groups = min(NThreadgroups, ntiles)
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @metal threads=(ThreadsPerGroup,) groups=(rank_groups,) onesweep_direct_rank_kernel!(
                    codes,
                    ws,
                    Val(TileSize),
                    Val(ThreadsPerGroup),
                    Val(digit),
                )
                @metal threads=(256,) groups=(1,) onesweep_direct_prefix_kernel!(
                    ws.lookback,
                    ntiles,
                )
                @metal threads=(ThreadsPerGroup,) groups=(ntiles,) onesweep_direct_scatter_kernel!(
                    codes,
                    ws,
                    Val(TileSize),
                    Val(digit),
                )
            end

            # Odd pass counts leave the final keys in the workspace buffer.
            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))
            return nothing
        end

        function UnsignedRadixSorts.radix_sort!(codes :: CodeV, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, TileSize, NThreadgroups, ThreadsPerGroup}
            Workspace = OnesweepWorkspace(MtlVector{$KeyT})
            return radix_sort!(codes, Workspace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
        end

        function UnsignedRadixSorts.radix_sortperm!(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, WorkspaceKeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, TileSize, NThreadgroups, ThreadsPerGroup}
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

            # Launch the same three-dispatch pass for paired key/permutation
            # sorting. The rank phase is shared; only the final scatter writes
            # the permutation value alongside its key.
            rank_groups = min(NThreadgroups, ntiles)
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @metal threads=(ThreadsPerGroup,) groups=(rank_groups,) onesweep_direct_rank_kernel!(
                    codes,
                    ws,
                    Val(TileSize),
                    Val(ThreadsPerGroup),
                    Val(digit),
                )
                @metal threads=(256,) groups=(1,) onesweep_direct_prefix_kernel!(
                    ws.lookback,
                    ntiles,
                )
                @metal threads=(ThreadsPerGroup,) groups=(ntiles,) onesweep_direct_perm_scatter_kernel!(
                    codes,
                    ws,
                    Val(TileSize),
                    Val(digit),
                )
            end

            # Odd pass counts leave the final keys in the workspace buffer;
            # the returned permutation side must match that final key side.
            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))
            return $(isodd(NPasses) ? :(ws.perms[2]) : :(ws.perms[1]))
        end

        function UnsignedRadixSorts.radix_sortperm!(codes :: CodeV, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, TileSize, NThreadgroups, ThreadsPerGroup}
            Workspace = OnesweepWorkspace(MtlVector{$KeyT})
            return radix_sortperm!(codes, Workspace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
        end
    end
end
