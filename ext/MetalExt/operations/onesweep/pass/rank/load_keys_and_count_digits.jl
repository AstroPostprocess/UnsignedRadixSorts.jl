"""
    _load_keys_and_count_digits!(src, local_counts, rangemin, tile_len, ::Val{Pass})

Count the radix digit histogram for the current Metal tile.

Each lane walks a strided subset of the tile, extracts the 8-bit digit with
`_radix_bucket(src[i], Pass)`, and atomically increments the threadgroup-local
bucket count.

CUB parallel: this covers the count-producing part of `LoadKeys` plus
`BlockRadixRankT::RankKeys(... CountsCallback(...))`.

# Parameters

- `src`: Active Metal source key buffer for this pass.
- `local_counts`: Threadgroup-memory bucket histogram updated in-place.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _load_keys_and_count_digits! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _load_keys_and_count_digits!(src :: KeyV, local_counts :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, SharedV <: MtlDeviceVector{UInt32}, Pass}
            lane_id = Int(Metal.thread_position_in_threadgroup().x)
            nlanes = Int(Metal.threads_per_threadgroup().x)
            local_i = lane_id
            while local_i <= tile_len
                i = rangemin + local_i - 1
                @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                Metal.atomic_fetch_add_explicit(pointer(local_counts, bucket), UInt32(1))
                local_i += nlanes
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            return nothing
        end
    end
end
