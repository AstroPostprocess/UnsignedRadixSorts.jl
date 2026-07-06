"""
    _load_keys!(src, keys, rangemin, tile_len, ::Val{TileSize})

Load one tile into the worker's cached-key temporary buffer.

CUB `LoadKeys(block_idx * TILE_ITEMS, keys)` fills
`bit_ordered_type keys[ITEMS_PER_THREAD]`. The CPU path stores the same tile in
a flat per-worker temporary slice; later `RankKeys` and `ScatterKeysShared`
helpers consume that cached tile instead of reloading `src`.

CUB reference: `AgentRadixSortOnesweep::LoadKeys`.

# Parameters

- `src`: Active source key buffer for this radix pass.
- `keys`: Per-worker cached-key temporary buffer.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.

# Returns

The same `keys` temporary buffer, now holding this worker's loaded tile.
"""
function _load_keys! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _load_keys!(src :: KeyV, keys :: KeyV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}) where {KeyV <: Vector{$KeyT}, TileSize}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)

            @inbounds for local_j in 1:tile_len
                keys[tile_base + local_j] = src[rangemin + local_j - 1]
            end

            return keys
        end
    end
end
