# Toolbox

################### Pass metadata ###################

## Return the number of 8-bit radix passes for an unsigned key type.
@inline _npasses(:: Type{KeyT}) where {KeyT <: Unsigned} = sizeof(KeyT)

## Return the number of 8-bit radix passes for a onesweep workspace.
@inline _npasses(:: OnesweepWorkspace{KeyT}) where {KeyT <: Unsigned} = _npasses(KeyT)

################### Bucket index helpers ###################

## Return the dense 1-based worker id for the current default thread-pool thread.
@inline function _worker_id()
    default_tids = Base.Threads.threadpooltids(:default)
    return Base.Threads.threadid() - first(default_tids) + 1
end

## Return the flattened base for a 256-bucket block.
@inline function _bucket_block_base(block_id :: Int)
    return 256 * (block_id - 1)
end

## Return the flattened index inside a 256-bucket block.
@inline function _bucket_block_index(block_id :: Int, bucket :: Int)
    return _bucket_block_base(block_id) + bucket
end

## Return the flattened index inside a 256-bucket block for a UInt64 bucket.
@inline function _bucket_block_index(block_id :: Int, bucket :: UInt64)
    return _bucket_block_base(block_id) + Int(bucket)
end

## Return the flattened bucket-offset base for a runtime pass.
@inline function _pass_bucket_base(pass :: Int)
    return _bucket_block_base(pass)
end

## Return the flattened all-pass bucket-offset index.
@inline function _bucket_offsets_index(pass :: Int, bucket :: Int)
    return _bucket_block_index(pass, bucket)
end

## Return the flattened all-pass bucket-offset index for a UInt64 bucket.
@inline function _bucket_offsets_index(pass :: Int, bucket :: UInt64)
    return _bucket_block_index(pass, bucket)
end

## Return the flattened per-worker prepass histogram index.
@inline function _prepass_counts_index(worker_id :: Int, pass :: Int, bucket :: Int, npasses :: Int)
    return 256 * npasses * (worker_id - 1) + _bucket_offsets_index(pass, bucket)
end

## Return the flattened per-worker prepass histogram index for a UInt64 bucket.
@inline function _prepass_counts_index(worker_id :: Int, pass :: Int, bucket :: UInt64, npasses :: Int)
    return 256 * npasses * (worker_id - 1) + _bucket_offsets_index(pass, bucket)
end

## Return the flattened per-worker bucket scratch base.
@inline function _worker_bucket_base(worker_id :: Int)
    return _bucket_block_base(worker_id)
end

## Return the flattened per-worker bucket scratch index.
@inline function _worker_bucket_index(worker_id :: Int, bucket :: Int)
    return _bucket_block_index(worker_id, bucket)
end

## Return the flattened per-worker bucket scratch index for a UInt64 bucket.
@inline function _worker_bucket_index(worker_id :: Int, bucket :: UInt64)
    return _bucket_block_index(worker_id, bucket)
end

################### Lookback table indexing ###################

## Return the flattened lookback-table index.
@inline function _lookback_index(tile_id :: Int, bucket :: Int)
    return 256 * tile_id + bucket
end

################### Lookback entry encoding ###################

## Pack a local count as a partial lookback entry.
@inline function _partial_entry(count :: UInt64)
    return count | (UInt64(1) << 62)
end

## Pack a prefix count as a global lookback entry.
@inline function _global_entry(count :: UInt64)
    return count | (UInt64(2) << 62)
end

## Extract the count payload from a lookback entry.
@inline function _entry_count(entry :: UInt64)
    return entry & ((UInt64(1) << 62) - UInt64(1))
end

## Return whether a lookback entry is partial.
@inline function _is_partial_entry(entry::UInt64)
    return (entry >> 62) == UInt64(1)
end

## Return whether a lookback entry is global.
@inline function _is_global_entry(entry::UInt64)
    return (entry >> 62) == UInt64(2)
end
