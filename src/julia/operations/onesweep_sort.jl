

function onesweep_sort!(codes :: Vector{UInt64}, ws :: OnesweepWorkspace{UInt64, Vector{UInt64}}, :: Val{NWorkers} = Val(nthreads()), :: Val{TileSize} = Val(4096)) where {NWorkers, TileSize}
    # Nelems: number of elements that need to be sorted
    nelems = length(codes)
    # Number of data tiles
    ntiles = cld(nelems, TileSize)

    # Initialize workspace
    initialize_workspace!(ws, nelems, ntiles, Val(NWorkers), Val(TileSize))

    # Onesweep passes, 8 bits per pass for UInt64
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(1))
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(2))
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(3))
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(4))
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(5))
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(6))
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(7))
    onesweep_pass!(codes,  ws, Val(NWorkers), Val(TileSize), Val(8))

    return nothing
end