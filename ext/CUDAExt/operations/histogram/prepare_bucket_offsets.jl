## CUB/CUDA vocabulary for this CUDA code:
##
## - CUDA block            ~= CUDA block
## - CUDA thread           ~= CUDA thread
## - threads=(...)         ~= CUDA blockDim.x
## - blocks=(...)          ~= CUDA gridDim.x
## - CuStaticSharedArray   ~= CUDA __shared__ memory
##
## The policy tuple below mirrors the CUB histogram policy fields needed here:
##
## - KeyT:    key type handled by this specialized method.
## - NPasses: number of 8-bit radix passes for KeyT.
## - NParts:  number of private histogram slices per (pass, bucket).
##
## NParts is intentionally a compile-time policy, not a caller argument. CUB uses
## a NUM_PARTS policy to reduce shared-memory atomic conflicts: a bucket is split
## into NParts counters, each thread chooses one part, and the parts are reduced
## before flushing to global memory.
for (KeyT, NPasses, NParts) in (
        (UInt8,    1,   8),
        (UInt16,   2,   8),
        (UInt32,   4,   8),
        (UInt64,   8,   4),
    )
    @eval begin
        # Compute CUB-style all-pass bucket starts for CUDA keys.
        #
        # ws.bucket_offsets is used like CUB's d_bins:
        #
        # 1. prepass_histogram_kernel! writes global counts for every
        #    (pass, bucket) into ws.bucket_offsets.
        # 2. bucket_offsets_exclusive_scan_kernel! converts those counts into
        #    1-based exclusive bucket starts for the later OneSweep pass.
        function UnsignedRadixSorts.prepare_bucket_offsets!(ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, codes :: CodeV, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(128)) where {CodeV <: CuVector{$KeyT}, WorkspaceKeyV <: CuVector{$KeyT}, OffsetV <: CuVector{UInt32}, NBlocks, ThreadsPerBlock}
            # Clear CUB's d_bins equivalent. After the histogram kernel this
            # buffer holds counts; after the scan kernel it holds bucket starts.
            fill!(ws.bucket_offsets, zero(UInt32))

            # CUDA mental model:
            #
            #   prepass_histogram_kernel<<<NBlocks, ThreadsPerBlock>>>
            #
            # Each block builds a shared-memory all-pass histogram,
            # then atomically flushes one count per non-empty (pass, bucket).
            CUDA.@cuda threads=ThreadsPerBlock blocks=NBlocks prepass_histogram_kernel!(ws, codes)

            # CUDA mental model:
            #
            #   bucket_offsets_exclusive_scan_kernel<<<NPasses, 256>>>
            #
            # This mirrors CUB's exclusive-sum kernel shape: one block
            # per radix pass, with 256 threads scanning that pass's 256 buckets.
            CUDA.@cuda threads=256 blocks=$NPasses bucket_offsets_exclusive_scan_kernel!(ws)

            return nothing
        end

        # CUB-style histogram kernel for one key type.
        #
        # Global output:
        #
        # - ws.bucket_offsets[pass, bucket] receives total counts.
        #
        # Block-local temporary:
        #
        # - bins[pass, bucket, part] in CUDA shared memory.
        #
        # The layout is flattened as:
        #
        #   idx = NParts * (256 * (pass - 1) + (bucket - 1)) + part
        #
        # where pass and bucket are 1-based Julia indices, and part is the
        # private histogram slice chosen from the CUDA thread id.
        @inline function prepass_histogram_kernel!(ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, codes :: CodeV) where {CodeV <: CuDeviceVector{$KeyT}, WorkspaceKeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}}
            nelems = length(codes)

            # CUDA equivalents:
            #
            #   block_id ~= blockIdx.x
            #   nblocks  ~= gridDim.x
            #   thread_id ~= threadIdx.x
            #   nthreads  ~= blockDim.x
            block_id = Int(CUDA.blockIdx().x)
            nblocks  = Int(CUDA.gridDim().x)
            thread_id = Int(CUDA.threadIdx().x)
            nthreads  = Int(CUDA.blockDim().x)

            # Shared histogram:
            #
            #   bins[pass, bucket, part]
            #
            # CUB uses multiple "parts" to reduce atomic contention when many
            # threads hit the same bucket. This is shared memory, not global
            # workspace memory.
            bins = CUDA.CuStaticSharedArray(UInt32, 256 * $NPasses * $NParts)

            # Cooperatively clear the whole shared histogram. Every thread clears
            # a strided subset, then the barrier ensures all bins are zero before
            # any thread starts accumulating counts.
            bin_idx = thread_id
            while bin_idx <= length(bins)
                @inbounds bins[bin_idx] = zero(UInt32)
                bin_idx += nthreads
            end
            CUDA.sync_threads()

            # Grid-stride loop over the input. This is the CUDA pattern:
            #
            #   i = blockIdx.x * blockDim.x + threadIdx.x
            #   stride = gridDim.x * blockDim.x
            #
            # adjusted for Julia's 1-based indices.
            i = (block_id - 1) * nthreads + thread_id
            stride = nblocks * nthreads

            # Pick the private histogram slice for this thread. Multiple threads
            # may share a part when nthreads > NParts, matching CUB's
            # conflict-reduction policy without allocating one private counter
            # per thread.
            part = ((thread_id - 1) % $NParts) + 1

            while i <= nelems
                @inbounds x = codes[i]
                @nexprs $NPasses pass -> begin
                    # Count this key in every radix pass. Global counts do not
                    # depend on input order, so CUB computes all passes once
                    # before launching the per-pass OneSweep scatter kernels.
                    bucket = UnsignedRadixSorts._radix_bucket(x, pass)

                    # Flatten bins[pass, bucket, part].
                    idx = $NParts * (256 * (pass - 1) + (bucket - 1)) + part

                    # High-frequency atomic, but only in shared memory. This is
                    # the CUB-style optimization over doing one global atomic per
                    # key/pass.
                    CUDA.atomic_add!(pointer(bins, idx), UInt32(1))
                end
                i += stride
            end
            CUDA.sync_threads()

            # Flush this block's shared histogram to global d_bins
            # (`ws.bucket_offsets`). Each thread owns a strided subset of the
            # flattened (pass, bucket) pairs, reduces the NParts counters for that
            # pair, and performs at most one global atomic add.
            bucket_idx = thread_id
            while bucket_idx <= 256 * $NPasses
                pass = fld(bucket_idx - 1, 256) + 1
                bucket = ((bucket_idx - 1) % 256) + 1

                # Reduce bins[pass, bucket, 1:NParts].
                count = zero(UInt32)
                @inbounds for part_idx in 1:$NParts
                    idx = $NParts * (256 * (pass - 1) + (bucket - 1)) + part_idx
                    count += bins[idx]
                end

                if count > zero(UInt32)
                    # Low-frequency global atomic:
                    #
                    #   one atomic per block/pass/bucket with non-zero count,
                    #   instead of one atomic per key/pass.
                    idx = UnsignedRadixSorts._bucket_offsets_index(pass, bucket)
                    CUDA.atomic_add!(pointer(ws.bucket_offsets, idx), count)
                end

                bucket_idx += nthreads
            end

            return nothing
        end 

        # Convert one pass of ws.bucket_offsets from counts to 1-based exclusive
        # bucket starts.
        #
        # CUB does this with a BlockScan over 256 buckets:
        #
        #   DeviceRadixSortExclusiveSumKernel<<<NPasses, 256>>>
        #
        # This CUDA version follows that shape. Each block handles one
        # pass, and each of its 256 threads owns one bucket. Warp scans plus a
        # scan of the eight warp totals implement the block-wide exclusive sum.
        @inline function bucket_offsets_exclusive_scan_kernel!(ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}) where {KeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}}
            pass = Int(CUDA.blockIdx().x)
            bucket = Int(CUDA.threadIdx().x)
            warp_threads = Int(CUDA.warpsize())
            warp_id = fld(bucket - 1, warp_threads)
            warp_lane_id = Int(CUDA.laneid())
            nwarps = 256 ÷ warp_threads
            full_mask = CUDA.FULL_MASK

            warp_totals = CUDA.CuStaticSharedArray(UInt32, 8)

            idx = UnsignedRadixSorts._bucket_offsets_index(pass, bucket)
            @inbounds count = ws.bucket_offsets[idx]
            inclusive = count

            offset = 1
            while offset < warp_threads
                addend = CUDA.shfl_up_sync(full_mask, inclusive, offset)
                if warp_lane_id > offset
                    inclusive += addend
                end
                offset <<= 1
            end

            if warp_lane_id == warp_threads
                @inbounds warp_totals[warp_id + 1] = inclusive
            end
            CUDA.sync_threads()

            if warp_id == 0
                warp_total = zero(UInt32)
                if warp_lane_id <= nwarps
                    @inbounds warp_total = warp_totals[warp_lane_id]
                end
                warp_inclusive = warp_total

                offset = 1
                while offset < warp_threads
                    addend = CUDA.shfl_up_sync(full_mask, warp_inclusive, offset)
                    if warp_lane_id > offset
                        warp_inclusive += addend
                    end
                    offset <<= 1
                end

                if warp_lane_id <= nwarps
                    @inbounds warp_totals[warp_lane_id] = warp_inclusive - warp_total
                end
            end
            CUDA.sync_threads()

            @inbounds warp_prefix = warp_totals[warp_id + 1]
            exclusive = warp_prefix + inclusive - count
            @inbounds ws.bucket_offsets[idx] = exclusive + one(UInt32)

            return nothing
        end
    end
end
