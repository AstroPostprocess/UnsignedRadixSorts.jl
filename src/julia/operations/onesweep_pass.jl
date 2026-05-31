
function onesweep_pass_kernel!(codes :: Vector{KeyT}, ws :: OnesweepWorkspace{KeyT, Vector{KeyT}}, :: Val{TileSize}, :: Val{Pass}) where {KeyT <: Unsigned, TileSize, Pass}
    lookback     = ws.lookback
    tile_counter = ws.tile_counter

    # Select source/output buffers and bucket-offset buffers for this pass.
    if isodd(Pass)
        src = codes
        dst = ws.dst
    else
        src = ws.dst
        dst = codes
    end

    nelems = length(src)
    ntiles = cld(nelems, TileSize)

    bucket_offsets = ws.bucket_offsets
    local_counts   = ws.local_counts
    local_offsets  = ws.local_offsets
    global_offsets = ws.global_offsets
    rank_cursors   = ws.rank_cursors
    local_ranks    = ws.local_ranks

    worker_id = _worker_id()
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
            idx = _worker_bucket_index(worker_id, b)
            local_counts[idx] = zero(UInt32)
        end

        @inbounds for i in rangemin:rangemax
            bucket = _radix_bucket(src[i], Pass)
            idx = _worker_bucket_index(worker_id, bucket)
            local_counts[idx] += one(UInt32)
        end

        # Third CUB-like step:
        # publish this tile's local counts as PARTIAL entries.
        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            count = local_counts[idx_wb]
            idx = _lookback_index(tile_id, bucket)
            Atomix.@atomic :release lookback[idx] = _partial_entry(count)
        end

        # Fourth CUB-like step:
        # compute 0-based tile-local exclusive digit offsets.
        running = zero(UInt32)

        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            local_offsets[idx_wb] = running
            running += local_counts[idx_wb]
        end

        # Fifth CUB-like step:
        # compute each element's 0-based tile-wide rank in bucket-sorted order.
        @inbounds for bucket in 1:256
            idx_wb = _worker_bucket_index(worker_id, bucket)
            rank_cursors[idx_wb] = local_offsets[idx_wb]
        end

        @inbounds for i in rangemin:rangemax
            bucket = _radix_bucket(src[i], Pass)
            idx_wb = _worker_bucket_index(worker_id, bucket)
            rank = rank_cursors[idx_wb]
            local_ranks[ranks_base + i - rangemin + 1] = rank
            rank_cursors[idx_wb] = rank + one(UInt32)
        end

        # Sixth CUB-like step:
        # look back over previous tiles and compute per-bucket global scatter offsets.
        @inbounds for bucket in 1:256
            previous = zero(UInt32)

            prev_tile = tile_id - 1
            while prev_tile >= 0
                idx = _lookback_index(prev_tile, bucket)

                entry = Atomix.@atomic :acquire lookback[idx]

                # Wait until the previous tile has published something.
                while entry == zero(UInt32)
                    entry = Atomix.@atomic :acquire lookback[idx]
                end

                previous += _entry_count(entry)

                if _is_global_entry(entry)
                    break
                end

                prev_tile -= 1
            end

            idx_wb = _worker_bucket_index(worker_id, bucket)
            local_count = local_counts[idx_wb]

            # This tile's actual scatter base for this bucket.
            idx_bo = _bucket_offsets_index(Pass, bucket)
            global_offsets[idx_wb] = bucket_offsets[idx_bo] + previous - local_offsets[idx_wb]

            # Update this tile's lookback entry from PARTIAL to GLOBAL.
            idx_l = _lookback_index(tile_id, bucket)
            Atomix.@atomic :release lookback[idx_l] = _global_entry(previous + local_count)
        end

        # Seventh CUB-like step:
        # scatter this tile using the bucket base and the 0-based tile-wide rank.
        @inbounds for i in rangemin:rangemax
            bucket = _radix_bucket(src[i], Pass)
            idx_wb = _worker_bucket_index(worker_id, bucket)
            rank_idx = ranks_base + i - rangemin + 1
            scatter_idx = global_offsets[idx_wb] + local_ranks[rank_idx]
            dst[Int(scatter_idx)] = src[i]
        end

    end
    
    return nothing
end
