"""
    _clear_rank_storage!(simd_offsets)

Clear the per-SIMD radix counters/cursors used by the next claimed tile.

This is the Metal counterpart of CUDA `_clear_rank_storage!`. The remaining
256-bin arrays are overwritten completely by the rank, scan, and lookback
stages and therefore do not need a separate zeroing pass.
"""
@inline function _clear_rank_storage!(simd_offsets :: SharedV) where {SharedV <: MtlDeviceVector{UInt32}}
    thread_id = Int(Metal.thread_position_in_threadgroup().x)
    nthreads = Int(Metal.threads_per_threadgroup().x)

    idx = thread_id
    while idx <= length(simd_offsets)
        @inbounds simd_offsets[idx] = zero(UInt32)
        idx += nthreads
    end

    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    return nothing
end
