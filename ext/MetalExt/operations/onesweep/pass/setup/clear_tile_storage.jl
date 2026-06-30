"""
    _clear_tile_storage!(local_counts, local_offsets, rank_cursors, global_offsets)

Clear the Metal threadgroup-local temporary storage used by the next claimed
tile.

Each lane clears a strided subset of the 256 radix buckets, then the
threadgroup synchronizes before counting starts. The arrays represent the
current threadgroup's bins, exclusive digit prefixes, rank cursors, and global
scatter bases.

CUB parallel: these arrays collectively stand in for the agent temporary
storage used for `bins`, `exclusive_digit_prefix`, rank cursors, and
`TempStorage_::global_offsets`.

# Parameters

- `local_counts`: Threadgroup-memory bucket counts; overwritten with zeros.
- `local_offsets`: Threadgroup-memory exclusive digit prefixes; overwritten with zeros.
- `rank_cursors`: Threadgroup-memory cursors used while assigning stable local ranks; overwritten with zeros.
- `global_offsets`: Threadgroup-memory global scatter bases; overwritten with zeros.
"""
@inline function _clear_tile_storage!(local_counts, local_offsets, rank_cursors, global_offsets)
    lane_id = Int(Metal.thread_position_in_threadgroup().x)
    nlanes = Int(Metal.threads_per_threadgroup().x)

    bucket = lane_id
    while bucket <= 256
        @inbounds begin
            local_counts[bucket] = zero(UInt32)
            local_offsets[bucket] = zero(UInt32)
            rank_cursors[bucket] = zero(UInt32)
            global_offsets[bucket] = zero(UInt32)
        end
        bucket += nlanes
    end
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    return nothing
end
