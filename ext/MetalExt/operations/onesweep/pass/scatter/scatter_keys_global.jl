using StaticArrays: MVector

"""
    _scatter_keys_shared!(keys_out, keys, ranks, tile_len, ::Val{TileSize}, ::Val{ThreadsPerGroup})

Reorder cached keys into tile-rank order in threadgroup memory.

CUB parallel: `ScatterKeysShared(keys, ranks)`.  CUDAExt uses this stage to
write each key to `s.keys_out[rank]` so later global scatter can read a
digit-sorted tile.

Metal note: this helper is kept for file-layout and naming parity with CUDAExt,
but the current Metal baseline does not call it.  The CUDA-style
`MVector ranks -> keys_out` shared permutation was the unstable path on Metal,
so `onesweep_pass.jl` uses re-read global scatter instead.

# Parameters

- `keys_out`: Threadgroup-memory key staging array.
- `keys`: Per-thread cached keys in SIMD-striped input order.
- `ranks`: Per-thread stable tile-local ranks.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{ThreadsPerGroup}`: Compile-time Metal threadgroup size.
"""
function _scatter_keys_shared! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_keys_shared!(keys_out :: SharedK, keys :: MVector{ItemsPerThread, $KeyT}, ranks :: MVector{ItemsPerThread, UInt32}, tile_len :: Int, :: Val{TileSize}, :: Val{ThreadsPerGroup}) where {ItemsPerThread, SharedK <: MtlDeviceVector{$KeyT}, TileSize, ThreadsPerGroup}
            # Get this Metal thread's SIMD-striped input coordinates.
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1

            # Each valid input item writes once to its tile-local sorted rank.
            item = 0
            while item < ItemsPerThread
                # Match the local_j mapping used by `_load_keys!` and RankKeys.
                local_j = simd_id * simd_threads * ItemsPerThread + item * simd_threads + lane_in_simd + 1
                if local_j <= tile_len
                    # `ranks` is 0-based like CUB's rank; Julia threadgroup-memory
                    # indexing is 1-based, so add one only at the array access.
                    @inbounds keys_out[Int(ranks[item + 1]) + 1] = keys[item + 1]
                end
                item += 1
            end

            # Ensure all sorted keys are visible before any global scatter reads them.
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            return nothing
        end
    end
end

"""
    _scatter_keys_global!(src, dst, global_offsets, ranks, rangemin, tile_len, ::Val{Pass})

Write source keys directly to final pass positions using saved tile-local ranks.

This keeps the CUDAExt helper name but intentionally uses Metal-specific re-read
staging.  Instead of reading a digit-sorted `keys_out` tile, threads walk the
original source tile, re-read `src[rangemin + local_i - 1]`, and combine the
lookback-resolved bucket base with `ranks[local_i]`.

CUB parallel: this replaces the dataflow of `ScatterKeysGlobal()` after
`ScatterKeysShared`.  The OneSweep index formula is unchanged:

    output_idx = global_bucket_base + stable_tile_rank

Why re-read?  The CUDA path depends on `MVector ranks -> keys_out[rank] = key`.
That shared permutation path has been observed to be unstable on Metal.jl.  The
re-read path costs one extra sequential source read but keeps ranks in explicit
threadgroup storage and avoids the problematic shared key permutation.

# Parameters

- `src`: Active source key buffer for this radix pass.
- `dst`: Active destination key buffer.
- `global_offsets`: Threadgroup per-bucket global scatter bases.
- `ranks`: Threadgroup tile-local ranks indexed by original tile-local input position.
- `rangemin`: First 1-based source index in the tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{Pass}`: Compile-time radix pass selector.
"""
function _scatter_keys_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _scatter_keys_global!(src :: KeyV, dst :: KeyV, global_offsets :: SharedV, ranks :: SharedV, rangemin :: Int, tile_len :: Int, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, SharedV <: MtlDeviceVector{UInt32}, Pass}
            # Walk the original tile in threadgroup-striped order.
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)

            local_i = thread_id
            while local_i <= tile_len
                # Re-read the source key instead of reading a CUB-style sorted
                # shared tile. `ranks[local_i]` already includes the local
                # bucket offset, so adding `global_offsets[bucket]` gives the
                # final 1-based output index for this pass.
                i = rangemin + local_i - 1
                @inbounds begin
                    key = src[i]
                    bucket = UnsignedRadixSorts._radix_bucket(key, Pass)
                    output_idx = global_offsets[bucket] + ranks[local_i]
                    dst[Int(output_idx)] = key
                end
                local_i += nthreads
            end

            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            return nothing
        end
    end
end
