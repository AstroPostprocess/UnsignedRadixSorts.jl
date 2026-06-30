"""
    _scatter_keys_global!(src, dst, global_offsets, local_ranks, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Scatter keys from the current tile to final pass positions.

This helper walks the original tile range, recomputes each key's bucket, reads
the bucket's `global_offsets` entry for the worker, adds the item rank from
`local_ranks`, and writes `src[i]` into `dst[Int(scatter_idx)]`.

CUB parallel: this is `ScatterKeysGlobal`; keys-only sorting has no value
scatter step.

# Parameters

- `src`: Active source key buffer for this pass.
- `dst`: Active destination key buffer for this pass.
- `global_offsets`: Per-worker global scatter bases produced by lookback resolution.
- `local_ranks`: Per-worker tile-local ranks.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid items in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _scatter_keys_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _scatter_keys_global!(src :: KeyV, dst :: KeyV, global_offsets :: OffsetV, local_ranks :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)
            rangemax = rangemin + tile_len - 1

            @inbounds for i in rangemin:rangemax
                rank_idx = tile_base + i - rangemin + 1
                bucket = _radix_bucket(src[i], Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                scatter_idx = global_offsets[idx_wb] + local_ranks[rank_idx]
                dst[Int(scatter_idx)] = src[i]
            end

            return nothing
        end
    end
end
