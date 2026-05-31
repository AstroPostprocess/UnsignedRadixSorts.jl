
function onesweep_sort!(codes :: Vector{UInt64}, :: Val{TileSize} = Val(4096)) where {TileSize}
    return onesweep_sort!(codes, _ONESWEEP_WORKSPACE_64, Val(TileSize))
end


function onesweep_sort!(codes :: Vector{UInt64}, ws :: OnesweepWorkspace{UInt64, Vector{UInt64}}, :: Val{TileSize} = Val(4096)) where {TileSize}
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
    @nexprs 8 digit -> begin
        reset_pass_workspace!(ws)
        @threads :static for _ in 1:nworkers
            onesweep_pass_kernel!(codes,  ws, Val(TileSize), Val(digit))
        end
    end

    return nothing
end
