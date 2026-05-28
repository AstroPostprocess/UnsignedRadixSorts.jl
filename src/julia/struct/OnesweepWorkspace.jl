struct OnesweepWorkspace{KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}, OffsetV <: AbstractVector{UInt64}}
    ## Workspace for data
    dst::KeyV

    ## Counter; length-1 vector, corresponds to CUB d_ctrs
    tile_counter::OffsetV

    ## Packed look-back table; flat vector, corresponds to CUB d_lookback
    ##
    ## Each entry uses the highest two bits as state:
    ##   entry == 0                 => EMPTY
    ##   01xxxxxx...xxxx            => PARTIAL local count
    ##   10xxxxxx...xxxx            => GLOBAL prefix count
    ##
    ## The lower 62 bits store the count.
    ## Length = 256 * ntile for an 8-bit radix pass.
    lookback::OffsetV

    ## Bucket offsets; correspond to CUB d_bins_in / d_bins_out.
    ##
    ## Both buffers store 1-based Julia output indices.
    ## bucket_offsets[1][bucket + 1] and bucket_offsets[2][bucket + 1]
    ## are the first Julia indices for `bucket`, depending on the pass.
    ##
    ## Odd passes read bucket_offsets[1] and write bucket_offsets[2].
    ## Even passes read bucket_offsets[2] and write bucket_offsets[1].
    ##
    ## Each buffer has length = 256 for an 8-bit radix pass.
    bucket_offsets::NTuple{2, OffsetV}

    function OnesweepWorkspace(::Type{KeyV}) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}}
        dst = KeyV(undef, 0)

        tile_counter = similar(dst, UInt64, 0)
        lookback = similar(dst, UInt64, 0)

        bucket_offsets_1 = similar(dst, UInt64, 0)
        bucket_offsets_2 = similar(dst, UInt64, 0)

        OffsetV = typeof(tile_counter)

        return new{KeyT, KeyV, OffsetV}(
            dst,
            tile_counter,
            lookback,
            (bucket_offsets_1, bucket_offsets_2),
        )
    end
end

function initialize_workspace!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems :: Int, ntiles :: Int) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}, OffsetV <: AbstractVector{UInt64}}
    resize!(ws.dst, nelems)
    resize!(ws.tile_counter, 1)
    resize!(ws.lookback, 256 * ntiles)

    resize!(ws.bucket_offsets[1], 256)
    resize!(ws.bucket_offsets[2], 256)

    ws.tile_counter[1] = UInt64(0)

    fill!(ws.lookback, UInt64(0))
    fill!(ws.bucket_offsets[1], UInt64(0))
    fill!(ws.bucket_offsets[2], UInt64(0))

    return nothing
end