
function onesweep_pass!(codes :: Vector{KeyT}, ws :: OnesweepWorkspace{KeyT, Vector{KeyT}}, :: Val{NWorkers}, :: Val{TileSize}, :: Val{Pass}) where {KeyT <: Unsigned, NWorkers, TileSize, Pass}
    # Initialize workspace for pass
    reset_pass_workspace!(ws)
    lookback     = ws.lookback
    tile_counter = ws.tile_counter

    # Select source/output buffers and bucket-offset buffers for this pass.
    if isodd(Pass)
        src = codes
        dst = ws.dst

        bucket_offsets_in  = ws.bucket_offsets[1]
        bucket_offsets_out = ws.bucket_offsets[2]
    else
        src = ws.dst
        dst = codes

        bucket_offsets_in  = ws.bucket_offsets[2]
        bucket_offsets_out = ws.bucket_offsets[1]
    end

    nelems = length(src)
    ntiles = cld(nelems, TileSize)

    local_counts   = ws.local_counts
    local_offsets  = ws.local_offsets
    global_offsets = ws.global_offsets
    local_ranks    = ws.local_ranks

    @threads for worker_id in 1:NWorkers
        # Per-worker local histogram slice
        bucket_base = 256 * (worker_id - 1)
        ranks_base = TileSize * (worker_id - 1)

        while true
            # First CUB-like step:
            # dynamically claim tile ids from the global counter.
            new = Atomix.@atomic tile_counter[1] += one(UInt32)
            tile_id = Int(new - UInt32(1))
            tile_id < ntiles || break

            # Convert the 0-based tile id to a 1-based Julia range.
            rangemin = tile_id * TileSize + 1
            rangemax = min(rangemin + TileSize - 1, nelems)

            # Second CUB-like step:
            # compute this tile's local radix histogram.
            @inbounds for b in 1:256
                local_counts[bucket_base + b] = zero(UInt32)
            end

            @inbounds for i in rangemin:rangemax
                bucket = _radix_bucket(src[i], Val(Pass))
                local_counts[bucket_base + bucket] += one(UInt32)
            end

            # Third CUB-like step:
            # publish this tile's local counts as PARTIAL entries.
            @inbounds for bucket in 1:256
                count = local_counts[bucket_base + bucket]
                idx = _lookback_index(tile_id, bucket)
                lookback[idx] = _partial_entry(count)
            end

            # Fourth CUB-like step:
            # compute tile-local exclusive digit offsets.
            running = zero(UInt32)

            @inbounds for bucket in 1:256
                local_offsets[bucket_base + bucket] = running
                running += local_counts[bucket_base + bucket]
            end

            # Fifth CUB-like step:
            # look back over previous tiles and compute per-bucket global scatter offsets.
            @inbounds for bucket in 1:256
                previous = zero(UInt32)

                prev_tile = tile_id - 1
                while prev_tile >= 0
                    idx = _lookback_index(prev_tile, bucket)

                    entry = lookback[idx]

                    # Wait until the previous tile has published something.
                    while entry == zero(UInt32)
                        entry = lookback[idx]
                    end

                    previous += _entry_count(entry)

                    if _is_global_entry(entry)
                        break
                    end

                    prev_tile -= 1
                end

                local_count = local_counts[bucket_base + bucket]

                # This tile's actual scatter base for this bucket.
                global_offsets[bucket_base + bucket] = bucket_offsets_in[bucket] + previous

                # Update this tile's lookback entry from PARTIAL to GLOBAL.
                lookback[_lookback_index(tile_id, bucket)] = _global_entry(previous + local_count)
            end

        end
    end
    return nothing
end

# Toolbox
@inline function _lookback_index(tile_id :: Int, bucket :: Int)
    return 256 * tile_id + bucket
end

@inline function _partial_entry(count :: UInt32)
    return count | (UInt32(1) << 30)
end

@inline function _global_entry(count :: UInt32)
    return count | (UInt32(2) << 30)
end

@inline function _entry_count(entry :: UInt32)
    return entry & ((UInt32(1) << 30) - UInt32(1))
end

@inline function _is_partial_entry(entry::UInt32)
    return (entry >> 30) == UInt32(1)
end

@inline function _is_global_entry(entry::UInt32)
    return (entry >> 30) == UInt32(2)
end