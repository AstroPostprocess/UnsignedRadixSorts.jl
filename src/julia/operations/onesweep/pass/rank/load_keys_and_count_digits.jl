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

"""
    _load_keys_and_count_digits!(src, local_counts, rangemin, tile_len, ::Val{Pass})

Count the radix digit histogram for the current tile.

`rangemin` and `tile_len` describe the claimed tile range in the active source
buffer. For each element, this helper extracts the 8-bit digit with
`_radix_bucket(src[i], Pass)`, maps that digit into the worker's private
histogram slot with `_worker_bucket_index`, and increments `local_counts[idx]`.

Legacy reference: pre-Process split stand-in for the
`RankKeys(... CountsCallback(...))` count-producing path. The current pass uses
`_load_keys!` plus `_rank_keys_early_counts!` instead.

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
