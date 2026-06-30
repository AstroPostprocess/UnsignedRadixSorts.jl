"""
    _claim_next_tile!(tile_counter, claimed_tile)

Claim the next OneSweep tile from the Metal global pass counter.

Lane 1 atomically increments `tile_counter[1]`, stores the claimed 0-based tile
id in threadgroup memory, and the threadgroup synchronizes so every lane
observes the same tile id.

CUB parallel: the agent constructor atomically claims `block_idx` before
running `Process()` for that tile.

# Parameters

- `tile_counter`: Length-one Metal pass counter mutated atomically.
- `claimed_tile`: Threadgroup-memory scalar used to broadcast the claimed tile id.
"""
@inline function _claim_next_tile!(tile_counter :: OffsetV, claimed_tile :: SharedV) where {OffsetV <: MtlDeviceVector{UInt32}, SharedV <: MtlDeviceVector{UInt32}}
    lane_id = Int(Metal.thread_position_in_threadgroup().x)

    if lane_id == 1
        @inbounds claimed_tile[1] = Metal.atomic_fetch_add_explicit(pointer(tile_counter, 1), UInt32(1))
    end
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    return Int(claimed_tile[1])
end
