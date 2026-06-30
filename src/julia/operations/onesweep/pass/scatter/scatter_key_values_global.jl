"""
    _scatter_key_values_global!(src, dst, perm_src, perm_dst, global_offsets, local_ranks, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Scatter keys and their permutation values to final pass positions.

This helper computes the same scatter index as `_scatter_keys_global!`, writes
`src[i]` to `dst`, and writes the paired permutation value `perm_src[i]` to
`perm_dst` at the same index.

CUB parallel: this covers `ScatterKeysGlobal` plus the value path represented
by `GatherScatterValues`.

# Parameters

- `src`: Active source key buffer for this pass.
- `dst`: Active destination key buffer for this pass.
- `perm_src`: Active source permutation buffer for this pass.
- `perm_dst`: Active destination permutation buffer for this pass.
- `global_offsets`: Per-worker global scatter bases produced by lookback resolution.
- `local_ranks`: Per-worker tile-local ranks.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _scatter_key_values_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _scatter_key_values_global!(src :: KeyV, dst :: KeyV, perm_src :: OffsetV, perm_dst :: OffsetV, global_offsets :: OffsetV, local_ranks :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)
            rangemax = rangemin + tile_len - 1

            @inbounds for i in rangemin:rangemax
                rank_idx = tile_base + i - rangemin + 1
                bucket = _radix_bucket(src[i], Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                scatter_idx = global_offsets[idx_wb] + local_ranks[rank_idx]
                dst[Int(scatter_idx)] = src[i]
                perm_dst[Int(scatter_idx)] = perm_src[i]
            end

            return nothing
        end
    end
end
