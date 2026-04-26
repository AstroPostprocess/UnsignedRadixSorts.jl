"""
    radix_offsets!(offsets :: MO, counts :: MC) where {TO <: Integer, TC <: Integer, MO <: AbstractMatrix{TO}, MC <: AbstractMatrix{TC}}

Compute the 1-based exclusive prefix offsets from a radix histogram and write the result in-place to `offsets`.

For each bucket index `i`, `offsets[i]` stores the first output position assigned to that bucket during the scatter step of a radix-sort pass. The function assumes that `counts[i]` contains the number of elements assigned to bucket `i`, and transforms these bucket counts into starting write positions using Julia's 1-based indexing convention.

The chunk-local counts are summed by bucket and the resulting global bucket
starts are stored in the first column of `offsets`.

# Parameters

- `offsets :: MO`: Preallocated output matrix. The first column is overwritten with global bucket start positions.
- `counts :: MC`: Radix histogram count matrix, where each column stores one chunk's bucket counts.
"""
@inline function radix_offsets!(offsets :: MO, counts :: MC) where {TO <: Integer, TC <: Integer, MO <: AbstractMatrix{TO}, MC <: AbstractMatrix{TC}}

    pos = one(TO)
    nchunks = size(counts, 2)

    @inbounds for bucket in axes(counts, 1)
        bucket_pos = pos

        for cid in 1:nchunks
            offsets[bucket, cid] = bucket_pos
            bucket_pos += TO(counts[bucket, cid])
        end

        pos = bucket_pos
    end
    return nothing
end
