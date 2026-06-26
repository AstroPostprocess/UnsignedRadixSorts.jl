for (KeyT, NPasses) in (
        (UInt8,   1),
        (UInt16,  2),
        (UInt32,  4),
        (UInt64,  8)
    )
    @eval begin
        function UnsignedRadixSorts.onesweep_sort!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NThreadgroups} = Val(256), :: Val{ThreadsPerGroup} = Val(256)) where {KeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, TileSize, NThreadgroups, ThreadsPerGroup}
            # Nelems: number of elements that need to be sorted
            nelems = length(codes)
            nelems == 0 && return nothing

            # Number of data tiles
            ntiles = cld(nelems, TileSize)
            # Number of workers/threads -> NThreadgroups

            # Initialize only the base workspace; local pass scratch lives in
            # Metal threadgroup memory.
            initialize_base_workspace!(ws, nelems, ntiles)

            # Before the first pass: global histogram for bucket
            prepare_bucket_offsets!(ws, codes, Val(NThreadgroups), Val(ThreadsPerGroup))

            # Onesweep passes, 8 bits per pass for UInt64
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) onesweep_pass_kernel!(codes, ws, Val(TileSize), Val(digit))
            end
            Metal.synchronize()

            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))

            return nothing
        end

        function UnsignedRadixSorts.onesweep_sort!(codes :: KeyV, :: Val{TileSize} = Val(4096), :: Val{NThreadgroups} = Val(256), :: Val{ThreadsPerGroup} = Val(256)) where {KeyV <: MtlVector{$KeyT}, TileSize, NThreadgroups, ThreadsPerGroup}
            Workspace = OnesweepWorkspace(MtlVector{$KeyT})
            return onesweep_sort!(codes, Workspace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NThreadgroups} = Val(256), :: Val{ThreadsPerGroup} = Val(256)) where {KeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, TileSize, NThreadgroups, ThreadsPerGroup}
            # Nelems: number of elements that need to be sorted
            nelems = length(codes)
            nelems == 0 && return similar(codes, UInt32, 0)

            # Number of data tiles
            ntiles = cld(nelems, TileSize)
            # Number of workers/threads -> NThreadgroups

            # Initialize workspace
            initialize_perm_workspace!(ws, nelems, ntiles, Val(NThreadgroups), Val(ThreadsPerGroup), Val(TileSize))

            # Before the first pass: global histogram for bucket
            prepare_bucket_offsets!(ws, codes, Val(NThreadgroups), Val(ThreadsPerGroup))

            # Onesweep passes, 8 bits per pass for UInt64
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) onesweep_perm_pass_kernel!(codes, ws, Val(TileSize), Val(digit))
            end
            Metal.synchronize()

            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))

            return $(isodd(NPasses) ? :(ws.perms[2]) : :(ws.perms[1]))
        end

        function UnsignedRadixSorts.onesweep_sortperm!(codes :: KeyV, :: Val{TileSize} = Val(4096), :: Val{NThreadgroups} = Val(256), :: Val{ThreadsPerGroup} = Val(256)) where {KeyV <: MtlVector{$KeyT}, TileSize, NThreadgroups, ThreadsPerGroup}
            Workspace = OnesweepWorkspace(MtlVector{$KeyT})
            return onesweep_sortperm!(codes, Workspace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
        end

    end
end
