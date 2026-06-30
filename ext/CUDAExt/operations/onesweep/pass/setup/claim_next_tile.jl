"""
    _claim_next_tile!(tile_counter, claimed_tile)

Claim the next OneSweep tile from the CUDA global pass counter.

Thread 1 atomically increments `tile_counter[1]`, stores the claimed 0-based
tile id in shared memory, and the block synchronizes so every thread observes
the same tile id.

CUB parallel: the agent constructor atomically claims `block_idx` before
running `Process()` for that tile.

# Parameters

- `tile_counter`: Length-one CUDA pass counter mutated atomically.
- `claimed_tile`: Shared-memory scalar used to broadcast the claimed tile id.
"""
@inline function _claim_next_tile!(tile_counter :: OffsetV, claimed_tile :: SharedV) where {OffsetV <: CuDeviceVector{UInt32}, SharedV <: CuDeviceVector{UInt32}}
    thread_id = Int(CUDA.threadIdx().x)

    if thread_id == 1
        @inbounds claimed_tile[1] = CUDA.atomic_add!(pointer(tile_counter, 1), UInt32(1))
    end
    CUDA.sync_threads()

    return Int(claimed_tile[1])
end
