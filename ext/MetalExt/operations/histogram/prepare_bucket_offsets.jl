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
## into NParts counters, each thread chooses one part, and the parts are reduced
## before flushing to global memory. Metal uses the same layout with smaller
## UInt32/UInt64 NParts values to stay below Apple's 32 KiB threadgroup-memory
## limit.

for (KeyT, NPasses, NParts) in (
        (UInt8,    1,   8),
        (UInt16,   2,   8),
        (UInt32,   4,   4),
        (UInt64,   8,   2),
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
        function UnsignedRadixSorts.prepare_bucket_offsets!(ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, codes :: CodeV, :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, WorkspaceKeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, NThreadgroups, ThreadsPerGroup}
            # Clear CUB's d_bins equivalent. After the histogram kernel this
            # buffer holds counts; after the scan kernel it holds bucket starts.
            fill!(ws.bucket_offsets, zero(UInt32))

            # CUDA mental model:
            #
            #   prepass_histogram_kernel<<<NThreadgroups, ThreadsPerGroup>>>
            #
            # Each threadgroup builds a threadgroup-memory all-pass histogram,
            # then atomically flushes one count per non-empty (pass, bucket).
            @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) prepass_histogram_kernel!(ws, codes)

            # CUDA mental model:
            #
            #   bucket_offsets_exclusive_scan_kernel<<<NPasses, 256>>>
            #
            # This mirrors CUB's exclusive-sum kernel shape: one threadgroup
            # per radix pass, with 256 threads scanning that pass's 256 buckets.
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
        # private histogram slice chosen from the Metal thread id.
        @inline function prepass_histogram_kernel!(ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, codes :: CodeV) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}}
            nelems = length(codes)

            # CUDA equivalents:
            #
            #   group_id  ~= blockIdx.x
            #   ngroups   ~= gridDim.x
            #   thread_id ~= threadIdx.x
            #   nthreads  ~= blockDim.x
            group_id = Int(Metal.threadgroup_position_in_grid().x)
            ngroups = Int(Metal.threadgroups_per_grid().x)
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)

            # Threadgroup histogram:
            #
            #   bins[pass, bucket, part]
            #
            # CUB uses multiple "parts" to reduce atomic contention when many
            # threads hit the same bucket. This is threadgroup memory, not
            # global workspace memory.
            bins = Metal.MtlThreadGroupArray(UInt32, 256 * $NPasses * $NParts)

            # Cooperatively clear the whole threadgroup histogram. Every thread
            # clears a strided subset, then the barrier ensures all bins are
            # zero before any thread starts accumulating counts.
            bin_idx = thread_id
            while bin_idx <= length(bins)
                @inbounds bins[bin_idx] = zero(UInt32)
                bin_idx += nthreads
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # Grid-stride loop over the input. This is the CUDA pattern:
            #
            #   i = blockIdx.x * blockDim.x + threadIdx.x
            #   stride = gridDim.x * blockDim.x
            #
            # adjusted for Julia's 1-based indices and Metal threadgroup names.
            i = (group_id - 1) * nthreads + thread_id
            stride = ngroups * nthreads

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

                    # High-frequency atomic, but only in threadgroup memory.
                    # This is the CUB-style optimization over doing one global
                    # atomic per key/pass.
                    Metal.atomic_fetch_add_explicit(pointer(bins, idx), UInt32(1))
                end
                i += stride
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            # Flush this threadgroup's histogram to global d_bins
            # (`ws.bucket_offsets`). Each thread owns a strided subset of the
            # flattened (pass, bucket) pairs, reduces the NParts counters for
            # that pair, and performs at most one global atomic add.
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
                    #   one atomic per threadgroup/pass/bucket with non-zero
                    #   count, instead of one atomic per key/pass.
                    idx = UnsignedRadixSorts._bucket_offsets_index(pass, bucket)
                    Metal.atomic_fetch_add_explicit(pointer(ws.bucket_offsets, idx), count)
                end
                bucket_idx += nthreads
            end

            return nothing
        end

        # Convert one pass of ws.bucket_offsets from counts to 1-based
        # exclusive bucket starts.
        #
        # CUB does this with a BlockScan over 256 buckets:
        #
        #   DeviceRadixSortExclusiveSumKernel<<<NPasses, 256>>>
        #
        # This Metal version follows that shape. Each threadgroup handles one
        # pass, and each of its 256 threads owns one bucket. SIMD scans plus a
        # scan of the eight SIMD-group totals implement the threadgroup-wide
        # exclusive sum.
        @inline function bucket_offsets_exclusive_scan_kernel!(ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}}
            pass = Int(Metal.threadgroup_position_in_grid().x)
            bucket = Int(Metal.thread_position_in_threadgroup().x)
            simd_threads = Int(Metal.threads_per_simdgroup())
            simd_id = Int(Metal.simdgroup_index_in_threadgroup()) - 1
            lane_in_simd = Int(Metal.thread_index_in_simdgroup()) - 1
            nsimdgroups = 256 ÷ 32
            simd_totals = Metal.MtlThreadGroupArray(UInt32, 8)

            idx = UnsignedRadixSorts._bucket_offsets_index(pass, bucket)
            @inbounds count = ws.bucket_offsets[idx]

            inclusive = count
            offset = 1
            while offset < simd_threads
                addend = Metal.simd_shuffle_up(inclusive, offset)
                if lane_in_simd >= offset
                    inclusive += addend
                end
                offset <<= 1
            end

            if lane_in_simd == simd_threads - 1
                @inbounds simd_totals[simd_id + 1] = inclusive
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            if simd_id == 0
                simd_total = lane_in_simd < nsimdgroups ? simd_totals[lane_in_simd + 1] : zero(UInt32)
                simd_inclusive = simd_total
                offset = 1
                while offset < simd_threads
                    addend = Metal.simd_shuffle_up(simd_inclusive, offset)
                    if lane_in_simd >= offset
                        simd_inclusive += addend
                    end
                    offset <<= 1
                end
                if lane_in_simd < nsimdgroups
                    @inbounds simd_totals[lane_in_simd + 1] = simd_inclusive - simd_total
                end
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            @inbounds simd_prefix = simd_totals[simd_id + 1]
            exclusive = simd_prefix + inclusive - count
            @inbounds ws.bucket_offsets[idx] = exclusive + one(UInt32)

            return nothing
        end
    end
end
