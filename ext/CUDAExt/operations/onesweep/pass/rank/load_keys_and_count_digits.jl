"""
    _load_keys_and_count_digits!(src, local_counts, rangemin, tile_len, ::Val{Pass})

Count the radix digit histogram for the current CUDA tile.

Each thread walks a strided subset of the tile, extracts the 8-bit digit with
`_radix_bucket(src[i], Pass)`, and atomically increments the block-local bucket
count in shared memory.

CUB parallel: this covers the count-producing part of `LoadKeys` plus
`BlockRadixRankT::RankKeys(... CountsCallback(...))`.

# Parameters

- `src`: Active CUDA source key buffer for this pass.
- `local_counts`: Shared-memory bucket histogram updated in-place.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _load_keys_and_count_digits! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _load_keys_and_count_digits!(src :: KeyV, local_counts :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{Pass}) where {KeyV <: CuDeviceVector{$KeyT}, SharedV <: CuDeviceVector{UInt32}, Pass}
            thread_id = Int(CUDA.threadIdx().x)
            nthreads = Int(CUDA.blockDim().x)
            local_i = thread_id
            while local_i <= tile_len
                i = rangemin + local_i - 1
                @inbounds bucket = UnsignedRadixSorts._radix_bucket(src[i], Pass)
                CUDA.atomic_add!(pointer(local_counts, bucket), UInt32(1))
                local_i += nthreads
            end
            CUDA.sync_threads()

            return nothing
        end
    end
end
