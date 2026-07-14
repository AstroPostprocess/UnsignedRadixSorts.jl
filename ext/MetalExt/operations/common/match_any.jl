"""
    _match_any(digit, valid, scratch, simd_id, lane_in_simd, simd_threads)

Return the peer mask of lanes in the calling SIMD group that hold the same
radix digit.

Metal.jl does not expose `simd_ballot`, but Apple AIR provides the same native
SIMD vote used by Metal Shading Language. Eight bit-plane ballots identify all
lanes with the caller's 8-bit radix digit without threadgroup scratch or SIMD
barriers. The unused scratch arguments preserve the rank helper's backend
interface.
"""
@inline function _air_simd_ballot(predicate::Bool)
    return Metal.@typed_ccall(
        "air.simd_ballot.i64",
        llvmcall,
        UInt64,
        (Bool,),
        predicate,
    )
end

@inline function _air_simd_shuffle(value::UInt32, source_lane::UInt16)
    return Metal.@typed_ccall(
        "air.simd_shuffle.u.i32",
        llvmcall,
        UInt32,
        (UInt32, UInt16),
        value,
        source_lane,
    )
end

@inline function _match_any(
        digit :: UInt32,
        valid :: Bool,
        scratch :: SharedV,
        simd_id :: Int,
        lane_in_simd :: Int,
        simd_threads :: Int,
    ) where {SharedV <: MtlDeviceVector{UInt32}}
    peers = _air_simd_ballot(valid)
    bit = UInt32(0)
    while bit < UInt32(8)
        bit_is_one = ((digit >> bit) & UInt32(1)) != UInt32(0)
        ones = _air_simd_ballot(valid && bit_is_one)
        peers &= bit_is_one ? ones : ~ones
        bit += UInt32(1)
    end
    return valid ? UInt32(peers) : UInt32(0)
end

"""
    _broadcast_peer_leader(value, leader_lane, scratch, simd_id, lane_in_simd, simd_threads)

Broadcast a peer-group leader's cursor value within one SIMD group through
AIR's arbitrary-source SIMD shuffle. The unused scratch arguments preserve the
rank helper's backend interface.
"""
@inline function _broadcast_peer_leader(
        value :: UInt32,
        leader_lane :: LeaderT,
        scratch :: SharedV,
        simd_id :: Int,
        lane_in_simd :: Int,
        simd_threads :: Int,
    ) where {LeaderT <: Integer, SharedV <: MtlDeviceVector{UInt32}}
    return _air_simd_shuffle(value, UInt16(leader_lane))
end
