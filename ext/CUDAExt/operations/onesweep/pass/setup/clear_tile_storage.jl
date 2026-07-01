"""
    _clear_rank_storage!(warp_offsets, ::Val{NWarps})

Clear the per-warp radix counters/cursors used by the next claimed tile.

`warp_offsets` has `NWarps * 256` entries. It first stores per-warp bucket
counts and is later overwritten with per-warp bucket cursors.

CUB parallel: this is the reusable BlockRadixRank scratch inside
`TempStorage_::rank_temp_storage`; it is cleared for each claimed tile before
`RankKeys` builds per-warp histograms.

# Parameters

- `warp_offsets`: Shared-memory per-warp bucket storage.
- `::Val{NWarps}`: Compile-time number of warps in the CUDA block.
"""
@inline function _clear_rank_storage!(warp_offsets :: SharedV, :: Val{NWarps}) where {SharedV <: CuDeviceVector{UInt32}, NWarps}
    # Use the whole block to clear the flattened warp/bucket table.
    thread_id = Int(CUDA.threadIdx().x)
    nthreads = Int(CUDA.blockDim().x)

    # Flattened layout: warp_offsets[warp * 256 + bucket].
    idx = thread_id
    while idx <= NWarps * 256
        # Clear every warp/bucket counter. The following barrier makes the
        # storage safe for the count phase of `_rank_keys_early_counts!`.
        @inbounds warp_offsets[idx] = zero(UInt32)
        idx += nthreads
    end

    # Ensure no stale cursor/count remains before RankKeys starts.
    CUDA.sync_threads()
    return nothing
end
