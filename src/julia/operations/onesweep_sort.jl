"""
    onesweep_sort!(codes::Vector{KeyT}, ws::OnesweepWorkspace{KeyT, Vector{KeyT}}, ::Val{TileSize}=Val(4096)) where {KeyT <: Unsigned, TileSize}
    onesweep_sort!(codes::Vector{KeyT}, ::Val{TileSize}=Val(4096)) where {KeyT <: Unsigned, TileSize}

Sort `codes` in-place in ascending order using the threaded OneSweep 8-bit LSD radix sorter.

# Parameters

- `codes`: Vector of unsigned integer keys to sort in-place.
- `ws`: Preallocated OneSweep workspace reused for temporary buffers and pass state.
- `::Val{TileSize}`: Compile-time tile size used by each OneSweep pass.
"""
function onesweep_sort! end

"""
    onesweep_sortperm!(codes::Vector{KeyT}, ws::OnesweepWorkspace{KeyT, Vector{KeyT}}, ::Val{TileSize}=Val(4096)) where {KeyT <: Unsigned, TileSize}
    onesweep_sortperm!(codes::Vector{KeyT}, ::Val{TileSize}=Val(4096)) where {KeyT <: Unsigned, TileSize}

Sort `codes` in-place in ascending order using the threaded OneSweep 8-bit LSD radix sorter, and return the corresponding permutation indices.

# Parameters

- `codes`: Vector of unsigned integer keys to sort in-place.
- `ws`: Preallocated OneSweep workspace reused for temporary buffers, permutation buffers, and pass state.
- `::Val{TileSize}`: Compile-time tile size used by each OneSweep pass.

# Returns

The permutation buffer whose entries are 1-based source indices for the sorted keys.
"""
function onesweep_sortperm! end

for (KeyT, Workspace, NPasses) in (
        (UInt8,   :_ONESWEEP_WORKSPACE_8,   1),
        (UInt16,  :_ONESWEEP_WORKSPACE_16,  2),
        (UInt32,  :_ONESWEEP_WORKSPACE_32,  4),
        (UInt64,  :_ONESWEEP_WORKSPACE_64,  8),
        (UInt128, :_ONESWEEP_WORKSPACE_128, 16),
    )
    @eval begin
        function onesweep_sort!(codes :: Vector{$KeyT}, ws :: OnesweepWorkspace{$KeyT, Vector{$KeyT}}, :: Val{TileSize} = Val(4096)) where {TileSize}
            # Nelems: number of elements that need to be sorted
            nelems = length(codes)
            # Number of data tiles
            ntiles = cld(nelems, TileSize)
            # Number of workers/threads
            nworkers = nthreads()

            # Initialize workspace
            initialize_workspace!(ws, nelems, ntiles, Val(nworkers), Val(TileSize))

            # Before the first pass: global histogram for bucket
            prepare_bucket_offsets!(ws, codes)

            # Onesweep passes, 8 bits per pass for UInt64
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @threads :static for _ in 1:nworkers
                    onesweep_pass_kernel!(codes,  ws, Val(TileSize), Val(digit))
                end
            end

            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))

            return nothing
        end

        function onesweep_sort!(codes :: Vector{$KeyT}, :: Val{TileSize} = Val(4096)) where {TileSize}
            return onesweep_sort!(codes, $Workspace, Val(TileSize))
        end

        function onesweep_sortperm!(codes :: Vector{$KeyT}, ws :: OnesweepWorkspace{$KeyT, Vector{$KeyT}}, :: Val{TileSize} = Val(4096)) where {TileSize}
            # Nelems: number of elements that need to be sorted
            nelems = length(codes)
            # Number of data tiles
            ntiles = cld(nelems, TileSize)
            # Number of workers/threads
            nworkers = nthreads()

            # Initialize workspace
            initialize_perm_workspace!(ws, nelems, ntiles, Val(nworkers), Val(TileSize))

            # Before the first pass: global histogram for bucket
            prepare_bucket_offsets!(ws, codes)

            # Onesweep passes, 8 bits per pass for UInt64
            @nexprs $NPasses digit -> begin
                reset_pass_workspace!(ws)
                @threads :static for _ in 1:nworkers
                    onesweep_perm_pass_kernel!(codes,  ws, Val(TileSize), Val(digit))
                end
            end

            $(isodd(NPasses) ? :(copyto!(codes, ws.dst)) : :(nothing))

            return $(isodd(NPasses) ? :(ws.perms[2]) : :(ws.perms[1]))
        end

        function onesweep_sortperm!(codes :: Vector{$KeyT}, :: Val{TileSize} = Val(4096)) where {TileSize}
            return onesweep_sortperm!(codes, $Workspace, Val(TileSize))
        end

    end
end
