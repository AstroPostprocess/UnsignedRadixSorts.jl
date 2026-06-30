"""
    _claim_next_tile!(tile_counter::OffsetV) where {OffsetV <: Vector{UInt32}}

Claim the next OneSweep tile from the global pass counter.

`tile_counter[1]` is a 1-element `Vector{UInt32}` shared by CPU workers for the
current pass. The atomic addition returns the post-increment value, so this
helper subtracts one and returns a 0-based `tile_id`. The caller compares that
id with `ntiles`, then maps it to Julia's 1-based `rangemin:rangemax`.

CUB parallel: the agent constructor atomically claims `block_idx` before
running `Process()` for that tile.

# Parameters

- `tile_counter`: Length-one pass counter mutated atomically.
"""
@inline function _claim_next_tile!(tile_counter :: OffsetV) where {OffsetV <: Vector{UInt32}}
    new = Atomix.@atomic tile_counter[1] += one(UInt32)
    return Int(new - UInt32(1))
end
