function prepare_bucket_offsets!(ws::OnesweepWorkspace{UInt64, Vector{UInt64}}, codes::Vector{UInt64})
    nworkers = nthreads()
    nelems = length(codes)
    npasses = _npasses(ws)

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

            @nexprs 8 pass -> begin
                bucket = _radix_bucket(x, pass)
                idx = _prepass_counts_index(worker_id, pass, bucket, npasses)
                ws.prepass_counts[idx] += one(UInt32)
            end
        end
    end

    # launch 2 equivalent:
    # reduce worker counts + counts -> 1-based exclusive starts
    @threads :static for pass in 1:npasses
        running = one(UInt32)

        @inbounds for bucket in 1:256
            count = zero(UInt32)

            for worker_id in 1:nworkers
                idx = _prepass_counts_index(worker_id, pass, bucket, npasses)
                count += ws.prepass_counts[idx]
            end

            idx = _bucket_offsets_index(pass, bucket)
            ws.bucket_offsets[idx] = running
            running += count
        end
    end

    return nothing
end
