## CUB/CUDA vocabulary for this Metal code:
##
## - Metal threadgroup      ~= CUDA block
## - Metal thread           ~= CUDA thread
## - threads=(...)          ~= CUDA blockDim.x
## - groups=(...)           ~= CUDA gridDim.x
## - MtlThreadGroupArray    ~= CUDA __shared__ memory
##
## The policy tuple below mirrors the CUB histogram policy fields needed here:
##
## - KeyT:    key type handled by this specialized method.
## - NPasses: number of 8-bit radix passes for KeyT.
## - NParts:  number of private histogram slices per (pass, bucket).
##
## NParts is intentionally a compile-time policy, not a caller argument. CUB uses
## a NUM_PARTS policy to reduce shared-memory atomic conflicts: a bucket is split
## into NParts counters, each lane chooses one part, and the parts are reduced
## before flushing to global memory.
for (KeyT, NPasses, NParts) in (
        (UInt8,    1,   8),
        (UInt16,   2,   8),
        (UInt32,   4,   8),
        (UInt64,   8,   4),
    )
    @eval begin
        # Compute CUB-style all-pass bucket starts for Metal keys.
        #
        # ws.bucket_offsets is used like CUB's d_bins:
        #
        # 1. prepass_histogram_kernel! writes global counts for every
        #    (pass, bucket) into ws.bucket_offsets.
        # 2. bucket_offsets_exclusive_scan_kernel! converts those counts into
        #    1-based exclusive bucket starts for the later OneSweep pass.
        function UnsignedRadixSorts.prepare_bucket_offsets!(ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, codes :: KeyV, :: Val{NThreadgroups} = Val(256), :: Val{ThreadsPerGroup} = Val(128)) where {KeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, NThreadgroups, ThreadsPerGroup}
            # Clear CUB's d_bins equivalent. After the histogram kernel this
            # buffer holds counts; after the scan kernel it holds bucket starts.
            fill!(ws.bucket_offsets, zero(UInt32))

            # CUDA mental model:
            #
            #   prepass_histogram_kernel<<<NThreadgroups, ThreadsPerGroup>>>
            #
            # Each block/threadgroup builds a shared-memory all-pass histogram,
            # then atomically flushes one count per non-empty (pass, bucket).
            @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) prepass_histogram_kernel!(ws, codes)

            # CUDA mental model:
            #
            #   bucket_offsets_exclusive_scan_kernel<<<NPasses, 256>>>
            #
            # This mirrors CUB's exclusive-sum kernel shape: one block/threadgroup
            # per radix pass, with 256 lanes scanning that pass's 256 buckets.
            @metal threads=(256,) groups=($NPasses,) bucket_offsets_exclusive_scan_kernel!(ws)

            return nothing
        end

        # CUB-style histogram kernel for one key type.
        #
        # Global output:
        #
        # - ws.bucket_offsets[pass, bucket] receives total counts.
        #
        # Threadgroup-local temporary:
        #
        # - bins[pass, bucket, part] in Metal threadgroup memory.
        #
        # The layout is flattened as:
        #
        #   idx = NParts * (256 * (pass - 1) + (bucket - 1)) + part
        #
        # where pass and bucket are 1-based Julia indices, and part is the
        # private histogram slice chosen from the lane id.
        @inline function prepass_histogram_kernel!(ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, codes :: KeyV) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}}
            nelems = length(codes)

            # CUDA equivalents:
            #
            #   group_id ~= blockIdx.x + 1
            #   ngroups  ~= gridDim.x
            #   lane_id  ~= threadIdx.x + 1
            #   nlanes   ~= blockDim.x
            group_id = Int(Metal.threadgroup_position_in_grid().x)
            ngroups  = Int(Metal.threadgroups_per_grid().x)
            lane_id  = Int(Metal.thread_position_in_threadgroup().x)
            nlanes   = Int(Metal.threads_per_threadgroup().x)

            # Shared/threadgroup histogram:
            #
            #   bins[pass, bucket, part]
            #
            # CUB uses multiple "parts" to reduce atomic contention when many
            # lanes hit the same bucket. This is threadgroup memory, not global
            # workspace memory.
            bins = Metal.MtlThreadGroupArray(UInt32, 256 * $NPasses * $NParts)

            # Cooperatively clear the whole shared histogram. Every lane clears
            # a strided subset, then the barrier ensures all bins are zero before
            # any lane starts accumulating counts.
            bin_idx = lane_id
            while bin_idx <= length(bins)
                @inbounds bins[bin_idx] = zero(UInt32)
                bin_idx += nlanes
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # Grid-stride loop over the input. This is the CUDA pattern:
            #
            #   i = blockIdx.x * blockDim.x + threadIdx.x
            #   stride = gridDim.x * blockDim.x
            #
            # adjusted for Julia's 1-based indices.
            i = (group_id - 1) * nlanes + lane_id
            stride = ngroups * nlanes

            # Pick the private histogram slice for this lane. Multiple lanes may
            # share a part when nlanes > NParts, matching CUB's conflict-reduction
            # policy without allocating one private counter per lane.
            part = ((lane_id - 1) % $NParts) + 1

            while i <= nelems
                @inbounds x = codes[i]
                @nexprs $NPasses pass -> begin
                    # Count this key in every radix pass. Global counts do not
                    # depend on input order, so CUB computes all passes once
                    # before launching the per-pass OneSweep scatter kernels.
                    bucket = UnsignedRadixSorts._radix_bucket(x, pass)

                    # Flatten bins[pass, bucket, part].
                    idx = $NParts * (256 * (pass - 1) + (bucket - 1)) + part

                    # High-frequency atomic, but only in threadgroup/shared
                    # memory. This is the CUB-style optimization over doing one
                    # global atomic per key/pass.
                    Metal.atomic_fetch_add_explicit(pointer(bins, idx), UInt32(1))
                end
                i += stride
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # Flush this threadgroup's shared histogram to global d_bins
            # (`ws.bucket_offsets`). Each lane owns a strided subset of the
            # flattened (pass, bucket) pairs, reduces the NParts counters for that
            # pair, and performs at most one global atomic add.
            bucket_idx = lane_id
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
                    #   one atomic per threadgroup/pass/bucket with non-zero
                    #   count, instead of one atomic per key/pass.
                    idx = UnsignedRadixSorts._bucket_offsets_index(pass, bucket)
                    Metal.atomic_fetch_add_explicit(pointer(ws.bucket_offsets, idx), count)
                end

                bucket_idx += nlanes
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
        # This Metal version follows that shape. Each threadgroup handles one
        # pass, and each of its 256 lanes owns one bucket. A simple shared-memory
        # Hillis-Steele scan is used here; CUB's BlockScan is more optimized, but
        # the dataflow and launch shape are the same.
        @inline function bucket_offsets_exclusive_scan_kernel!(ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}}
            # group_id is the pass id because this kernel is launched with
            # groups=(NPasses,) and threads=(256,).
            pass = Int(Metal.threadgroup_position_in_grid().x)
            bucket = Int(Metal.thread_position_in_threadgroup().x)

            counts = Metal.MtlThreadGroupArray(UInt32, 256)

            # Load this pass's bucket counts into threadgroup memory. At this
            # point ws.bucket_offsets still holds histogram counts, not starts.
            idx = UnsignedRadixSorts._bucket_offsets_index(pass, bucket)
            @inbounds counts[bucket] = ws.bucket_offsets[idx]
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # Inclusive scan over counts[1:256]. This is the straightforward
            # shared-memory scan equivalent of CUB BlockScan, not its optimized
            # implementation.
            offset = 1
            while offset < 256
                addend = bucket > offset ? counts[bucket - offset] : zero(UInt32)

                # Ensure every lane has read the previous scan stage before any
                # lane writes this stage.
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
                @inbounds counts[bucket] += addend
                Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

                offset <<= 1
            end

            # Convert inclusive counts to 1-based exclusive starts. CUB's d_bins
            # are 0-based offsets; this Julia sorter stores output indices, so
            # bucket 1 starts at index 1.
            start = bucket == 1 ? one(UInt32) : counts[bucket - 1] + one(UInt32)
            @inbounds ws.bucket_offsets[idx] = start

            return nothing
        end
    end
end
