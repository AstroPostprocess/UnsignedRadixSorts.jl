"""
    _scatter_key_values_global!(keys_out, values_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Scatter CUB `s.keys_out` entries and paired permutation values.

This mirrors CUB `ScatterKeysGlobal()` and
`GatherScatterValues(ranks, bool_constant_v<KEYS_ONLY>)`: values are loaded in
original tile order and scattered to `s.values_out[rank]`; then staged keys and
values are written to their shared global positions in one pass over sorted
tile order.

# Parameters

- `keys_out`: Per-worker tile-ranked key staging buffer.
- `values_out`: Per-worker tile-ranked permutation value staging buffer.
- `keys`: Per-worker cached keys loaded by `_load_keys!`.
- `ranks`: Per-worker 0-based tile-local ranks.
- `dst`: Active destination key buffer for this pass.
- `perm_src`: Active source permutation buffer for this pass.
- `perm_dst`: Active destination permutation buffer for this pass.
- `global_offsets`: Per-worker global scatter bases produced by lookback resolution.
- `rangemin`: First 1-based source index in the claimed tile.
- `tile_len`: Number of valid keys in the tile.
- `::Val{TileSize}`: Compile-time tile size.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _scatter_key_values_global! end

for KeyT in (UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval begin
        @inline function _scatter_key_values_global!(keys_out :: KeyV, values_out :: OffsetV, keys :: KeyV, ranks :: OffsetV, dst :: KeyV, perm_src :: OffsetV, perm_dst :: OffsetV, global_offsets :: OffsetV, rangemin :: Int, tile_len :: Int, :: Val{TileSize}, :: Val{Pass}) where {KeyV <: Vector{$KeyT}, OffsetV <: Vector{UInt32}, TileSize, Pass}
            worker_id = _worker_id()
            tile_base = TileSize * (worker_id - 1)

            # -----------------------------------------------------
            # 1. LoadValues + ScatterValuesShared.
            #
            # Load permutation values in original tile order and stage them as
            # s.values_out[rank], using the same ranks as keys.
            @inbounds for local_j in 1:tile_len
                rank_idx = tile_base + local_j
                staged_idx = tile_base + Int(ranks[rank_idx]) + 1
                values_out[staged_idx] = perm_src[rangemin + local_j - 1]
            end

            # -----------------------------------------------------
            # 2. ScatterKeysGlobal + ScatterValuesGlobal.
            #
            # Both outputs use the same sorted key bucket and sorted_idx0, so
            # compute the global position once and write the paired key/value.
            @inbounds for sorted_idx0 in UInt32(0):UInt32(tile_len - 1)
                sorted_idx = tile_base + Int(sorted_idx0) + 1
                key = keys_out[sorted_idx]
                bucket = _radix_bucket(key, Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                scatter_idx = global_offsets[idx_wb] + sorted_idx0
                output_idx = Int(scatter_idx)
                dst[output_idx] = key
                perm_dst[output_idx] = values_out[sorted_idx]
            end

            return nothing
        end
    end
end
