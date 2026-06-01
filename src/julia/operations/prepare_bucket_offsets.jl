
"""
    prepare_bucket_offsets!(ws::OnesweepWorkspace{KeyT, Vector{KeyT}}, codes::Vector{KeyT}) where {KeyT <: Unsigned}

Compute all-pass radix histograms for `codes` and write 1-based global bucket start offsets into `ws`.

# Parameters

- `ws`: OneSweep workspace whose prepass count buffer and bucket offset buffer are overwritten.
- `codes`: Input vector of unsigned keys used to build the per-pass bucket offsets.
"""
function prepare_bucket_offsets! end


for (KeyT, NPasses) in (
        (UInt8,    1),
        (UInt16,   2),
        (UInt32,   4),
        (UInt64,   8),
        (UInt128, 16),
    )
    @eval begin
        function prepare_bucket_offsets!(ws :: OnesweepWorkspace{$KeyT, Vector{$KeyT}}, codes :: Vector{$KeyT})
            nworkers = nthreads()
            nelems = length(codes)

            # Initialization
            fill!(ws.prepass_counts, zero(UInt32))
            fill!(ws.bucket_offsets, zero(UInt32))

            # launch 1 equivalent:
            # all-pass histogram, per worker private counts
            @threads :static for worker_id in 1:nworkers
                lo = fld((worker_id - 1) * nelems, nworkers) + 1
                hi = fld(worker_id * nelems, nworkers)

                @inbounds for i in lo:hi
                    x = codes[i]

                    @nexprs $NPasses pass -> begin
                        bucket = _radix_bucket(x, pass)
                        idx = _prepass_counts_index(worker_id, pass, bucket, $NPasses)
                        ws.prepass_counts[idx] += one(UInt32)
                    end
                end
            end

            # launch 2 equivalent:
            # reduce worker counts + counts -> 1-based exclusive starts
            @threads :static for pass in 1:$NPasses
                running = one(UInt32)

                @inbounds for bucket in 1:256
                    count = zero(UInt32)

                    for worker_id in 1:nworkers
                        idx = _prepass_counts_index(worker_id, pass, bucket, $NPasses)
                        count += ws.prepass_counts[idx]
                    end

                    idx = _bucket_offsets_index(pass, bucket)
                    ws.bucket_offsets[idx] = running
                    running += count
                end
            end

            return nothing
        end
    end
end
