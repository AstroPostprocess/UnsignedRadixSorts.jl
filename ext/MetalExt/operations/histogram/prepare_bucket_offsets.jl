## CUB/CUDA vocabulary for this Metal code:
##
## - Metal threadgroup      ~= CUDA block
## - Metal thread           ~= CUDA thread
## - threads=(...)          ~= CUDA blockDim.x
## - groups=(...)           ~= CUDA gridDim.x
## - MtlThreadGroupArray    ~= CUDA __shared__ memory

for (KeyT, NPasses, NParts) in (
        (UInt8,    1,   8),
        (UInt16,   2,   8),
        (UInt32,   4,   4),
        (UInt64,   8,   2),
    )
    @eval begin
        function UnsignedRadixSorts.prepare_bucket_offsets!(ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, codes :: CodeV, :: Val{NThreadgroups} = Val(8), :: Val{ThreadsPerGroup} = Val(256)) where {CodeV <: MtlVector{$KeyT}, WorkspaceKeyV <: MtlVector{$KeyT}, OffsetV <: MtlVector{UInt32}, NThreadgroups, ThreadsPerGroup}
            fill!(ws.bucket_offsets, zero(UInt32))
            @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) prepass_histogram_kernel!(ws, codes)
            @metal threads=(256,) groups=($NPasses,) bucket_offsets_exclusive_scan_kernel!(ws)
            return nothing
        end

        @inline function prepass_histogram_kernel!(ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, codes :: CodeV) where {CodeV <: MtlDeviceVector{$KeyT}, WorkspaceKeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}}
            nelems = length(codes)
            group_id = Int(Metal.threadgroup_position_in_grid().x)
            ngroups = Int(Metal.threadgroups_per_grid().x)
            thread_id = Int(Metal.thread_position_in_threadgroup().x)
            nthreads = Int(Metal.threads_per_threadgroup().x)

            bins = Metal.MtlThreadGroupArray(UInt32, 256 * $NPasses * $NParts)

            bin_idx = thread_id
            while bin_idx <= length(bins)
                @inbounds bins[bin_idx] = zero(UInt32)
                bin_idx += nthreads
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            i = (group_id - 1) * nthreads + thread_id
            stride = ngroups * nthreads
            part = ((thread_id - 1) % $NParts) + 1

            while i <= nelems
                @inbounds x = codes[i]
                @nexprs $NPasses pass -> begin
                    bucket = UnsignedRadixSorts._radix_bucket(x, pass)
                    idx = $NParts * (256 * (pass - 1) + (bucket - 1)) + part
                    Metal.atomic_fetch_add_explicit(pointer(bins, idx), UInt32(1))
                end
                i += stride
            end
            Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            bucket_idx = thread_id
            while bucket_idx <= 256 * $NPasses
                pass = fld(bucket_idx - 1, 256) + 1
                bucket = ((bucket_idx - 1) % 256) + 1
                count = zero(UInt32)
                @inbounds for part_idx in 1:$NParts
                    idx = $NParts * (256 * (pass - 1) + (bucket - 1)) + part_idx
                    count += bins[idx]
                end
                if count > zero(UInt32)
                    idx = UnsignedRadixSorts._bucket_offsets_index(pass, bucket)
                    Metal.atomic_fetch_add_explicit(pointer(ws.bucket_offsets, idx), count)
                end
                bucket_idx += nthreads
            end

            return nothing
        end

        # CUDA-equivalent hierarchical 256-bin scan: SIMD scans plus a scan of
        # the eight SIMD-group totals. This replaces the old Hillis-Steele path.
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
