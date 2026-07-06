"""
    _scatter_keys_shared!(keys_out, keys, ranks, tile_len, ::Val{TileSize})

Reorder cached keys into CUB `s.keys_out` order.

CUB `ScatterKeysShared(keys, ranks)` writes each key to `s.keys_out[rank]`.
The CPU path stores the same rank-ordered tile in the worker's `keys_out`
temporary slice. `ranks` are CUB-style 0-based tile-local ranks; Julia indexing
adds one at the array access.

# Parameters

- `keys_out`: Per-worker tile-ranked key staging buffer.
- `keys`: Per-worker cached keys loaded by `_load_keys!`.
- `ranks`: Per-worker 0-based tile-local ranks.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
"""
function _scatter_keys_shared! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _scatter_keys_shared!(keys_out :: KeyV, keys :: KeyV, ranks :: OffsetV, tile_len :: Int, :: Val{TileSize}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, TileSize}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)

            @inbounds for local_j in 1:tile_len
                rank_idx = tile_base + local_j
                staged_idx = tile_base + Int(ranks[rank_idx]) + 1
                keys_out[staged_idx] = keys[rank_idx]
            end

            return nothing
        end
    end
end

"""
    _scatter_keys_global!(keys_out, dst, global_offsets, tile_len, ::Val{TileSize}, ::Val{Pass})

Write CUB `s.keys_out` entries to final pass positions.

This mirrors CUB `ScatterKeysGlobal()`: read the rank-ordered tile from
`s.keys_out`, recompute `Digit(key)`, add `s.global_offsets[Digit(key)]`, and
write to `d_keys_out`.

CUB reference: `ScatterKeysGlobal()`.

# Parameters

- `keys_out`: Per-worker tile-ranked key staging buffer.
- `dst`: Active destination key buffer for this pass.
- `global_offsets`: Per-worker global scatter bases produced by lookback resolution.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _scatter_keys_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _scatter_keys_global!(keys_out :: KeyV, dst :: KeyV, global_offsets :: OffsetV, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)

            @inbounds for sorted_idx0 in UInt32(0):UInt32(tile_len - 1)
                key = keys_out[tile_base + Int(sorted_idx0) + 1]
                bucket = _radix_bucket(key, Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                scatter_idx = global_offsets[idx_wb] + sorted_idx0
                dst[Int(scatter_idx)] = key
            end

            return nothing
        end
    end
end
