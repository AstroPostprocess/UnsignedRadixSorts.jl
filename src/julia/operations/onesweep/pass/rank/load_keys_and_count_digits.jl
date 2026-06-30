"""
    _load_keys_and_count_digits!(src, local_counts, rangemin, tile_len, ::Val{Pass})

Count the radix digit histogram for the current tile.

`rangemin` and `tile_len` describe the claimed tile range in the active source
buffer. For each element, this helper extracts the 8-bit digit with
`_radix_bucket(src[i], Pass)`, maps that digit into the worker's private
histogram slot with `_worker_bucket_index`, and increments `local_counts[idx]`.

CUB parallel: this covers the count-producing part of `LoadKeys` plus
`BlockRadixRankT::RankKeys(... CountsCallback(...))`.

# Parameters

- `src`: Active source key buffer for this pass.
- `local_counts`: Per-worker bucket histogram updated in-place.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _load_keys_and_count_digits! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _load_keys_and_count_digits!(src :: KeyV, local_counts :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{Pass}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, Pass}
            worker_id = _worker_id()
            rangemax = rangemin + tile_len - 1

            @inbounds for i in rangemin:rangemax
                bucket = _radix_bucket(src[i], Pass)
                idx = _worker_bucket_index(worker_id, bucket)
                local_counts[idx] += one(UInt32)
            end

            return nothing
        end
    end
end
