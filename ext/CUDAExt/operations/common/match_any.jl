"""
    _match_any(full_mask, digit, valid)

Return the active-lane mask for lanes whose 8-bit radix digit matches `digit`.

The implementation mirrors CCCL's small-label match path by intersecting one
warp ballot per digit bit. Invalid lanes are excluded from every peer group.

# Parameters

- `full_mask`: CUDA warp mask supplied to the synchronized vote operations.
- `digit`: Zero-based 8-bit radix digit for the current lane.
- `valid`: Whether the current lane contains a valid tile item.

# Returns

- A `UInt32` mask containing all active valid lanes with the same digit.
"""
@inline function _match_any(full_mask :: UInt32, digit :: UInt32, valid :: Bool)
    # Start with all valid lanes in the warp.
    active_mask = CUDA.vote_ballot_sync(full_mask, valid)
    peer_mask = active_mask

    # Intersect one ballot per digit bit to keep only lanes with matching bits.
    for bit in 0:7
        bit_is_set = ((digit >> bit) & UInt32(1)) != 0
        set_mask = CUDA.vote_ballot_sync(full_mask, valid && bit_is_set)
        peer_mask &= bit_is_set ? set_mask : (active_mask & ~set_mask)
    end

    # Invalid lanes must not participate in any peer group.
    return valid ? peer_mask : zero(UInt32)
end
