"""
    _scatter_key_values_global!(keys_out, values_out, keys, ranks, dst, perm_src, perm_dst, global_offsets, rangemin, tile_len, ::Val{TileSize}, ::Val{Pass})

Scatter CUB `s.keys_out` entries and paired permutation values.

This mirrors CUB `ScatterKeysGlobal()` followed by
`GatherScatterValues(ranks, bool_constant_v<KEYS_ONLY>)`: staged keys are
written globally first, values are loaded in original tile order, scattered to
`s.values_out[rank]`, and then written to `d_values_out` at the same global
positions as the sorted keys.

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
            # 1. ScatterKeysGlobal: s.keys_out -> d_keys_out.
            @inbounds for sorted_idx0 in UInt32(0):UInt32(tile_len - 1)
                key = keys_out[tile_base + Int(sorted_idx0) + 1]
                bucket = _radix_bucket(key, Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                scatter_idx = global_offsets[idx_wb] + sorted_idx0
                dst[Int(scatter_idx)] = key
            end

            # -----------------------------------------------------
            # 2. LoadValues + ScatterValuesShared.
            #
            # Load permutation values in original tile order and stage them as
            # s.values_out[rank], using the same ranks as keys.
            @inbounds for local_j in 1:tile_len
                rank_idx = tile_base + local_j
                staged_idx = tile_base + Int(ranks[rank_idx]) + 1
                values_out[staged_idx] = perm_src[rangemin + local_j - 1]
            end

            # -----------------------------------------------------
            # 3. ScatterValuesGlobal: s.values_out -> d_values_out.
            @inbounds for sorted_idx0 in UInt32(0):UInt32(tile_len - 1)
                key = keys_out[tile_base + Int(sorted_idx0) + 1]
                bucket = _radix_bucket(key, Pass)
                idx_wb = _worker_bucket_index(worker_id, bucket)
                scatter_idx = global_offsets[idx_wb] + sorted_idx0
                perm_dst[Int(scatter_idx)] = values_out[tile_base + Int(sorted_idx0) + 1]
            end

            return nothing
        end
    end
end

# Legacy pre-keys_out path retained for callers that still scatter keys and
# values directly from the source tile.
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
