"""
    _match_any(digit, valid, scratch, simd_id, lane_in_simd, simd_threads)

Return the peer mask of lanes in the calling SIMD group that hold the same
radix digit.

Metal.jl does not currently expose `simd_ballot`, so this implementation uses
one threadgroup-memory segment per SIMD group. Only lanes in the same SIMD
group access a segment, and both synchronization points therefore use
`simdgroup_barrier` rather than a full threadgroup barrier.

The trailing barrier is required. The same segment is reused by the following
leader-broadcast phase or by the next item, so no lane may overwrite its slot
until every lane in the SIMD group has completed the current peer scan.
"""
@inline function _match_any(
        digit :: UInt32,
        valid :: Bool,
        scratch :: SharedV,
        simd_id :: Int,
        lane_in_simd :: Int,
        simd_threads :: Int,
    ) where {SharedV <: MtlDeviceVector{UInt32}}
    base = simd_id * simd_threads
    slot = base + lane_in_simd + 1

    # Invalid tail lanes publish a sentinel that cannot equal any 8-bit radix
    # digit. They still execute the same SIMD barriers as valid lanes.
    @inbounds scratch[slot] = valid ? digit : typemax(UInt32)
    Metal.simdgroup_barrier(Metal.MemoryFlagThreadGroup)

    peer_mask = zero(UInt32)
    if valid
        src_lane = 0
        while src_lane < simd_threads
            @inbounds peer_digit = scratch[base + src_lane + 1]
            if peer_digit == digit
                peer_mask |= UInt32(1) << UInt32(src_lane)
            end
            src_lane += 1
        end
    end

    Metal.simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    return peer_mask
end

"""
    _broadcast_peer_leader(value, leader_lane, scratch, simd_id, lane_in_simd, simd_threads)

Broadcast a peer-group leader's cursor value within one SIMD group.

The caller passes `leader_lane == lane_in_simd` and `value == 0` for invalid
tail lanes. Consequently, every invalid lane writes and reads only its own
slot; it never consumes a stale slot belonging to a valid peer group.
"""
@inline function _broadcast_peer_leader(
        value :: UInt32,
        leader_lane :: LeaderT,
        scratch :: SharedV,
        simd_id :: Int,
        lane_in_simd :: Int,
        simd_threads :: Int,
    ) where {LeaderT <: Integer, SharedV <: MtlDeviceVector{UInt32}}
    leader = Int(leader_lane)
    base = simd_id * simd_threads

    if lane_in_simd == leader
        @inbounds scratch[base + lane_in_simd + 1] = value
    end
    Metal.simdgroup_barrier(Metal.MemoryFlagThreadGroup)

    @inbounds result = scratch[base + leader + 1]

    # The segment is immediately reused by the next `_match_any` call.
    Metal.simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    return result
end
