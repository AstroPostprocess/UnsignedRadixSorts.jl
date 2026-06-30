"""
    _clear_tile_storage!(local_counts, local_offsets, rank_cursors, global_offsets)

Clear the CUDA block-local temporary storage used by the next claimed tile.

Each thread clears a strided subset of the 256 radix buckets, then the block
synchronizes before counting starts. The arrays represent the current block's
bins, exclusive digit prefixes, rank cursors, and global scatter bases.

CUB parallel: these arrays collectively stand in for the agent temporary
storage used for `bins`, `exclusive_digit_prefix`, rank cursors, and
`TempStorage_::global_offsets`.

# Parameters

- `local_counts`: Shared-memory bucket counts; overwritten with zeros.
- `local_offsets`: Shared-memory exclusive digit prefixes; overwritten with zeros.
- `rank_cursors`: Shared-memory cursors used while assigning stable local ranks; overwritten with zeros.
- `global_offsets`: Shared-memory global scatter bases; overwritten with zeros.
"""
@inline function _clear_tile_storage!(local_counts :: SharedV, local_offsets :: SharedV, rank_cursors :: SharedV, global_offsets :: SharedV) where {SharedV <: CuDeviceVector{UInt32}}
    thread_id = Int(CUDA.threadIdx().x)
    nthreads = Int(CUDA.blockDim().x)

    bucket = thread_id
    while bucket <= 256
        @inbounds begin
            local_counts[bucket] = zero(UInt32)
            local_offsets[bucket] = zero(UInt32)
            rank_cursors[bucket] = zero(UInt32)
            global_offsets[bucket] = zero(UInt32)
        end
        bucket += nthreads
    end
    CUDA.sync_threads()

    return nothing
end
