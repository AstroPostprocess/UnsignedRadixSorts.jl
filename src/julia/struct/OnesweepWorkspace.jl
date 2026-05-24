struct OnesweepWorkspace{KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}, OffsetV <: AbstractVector{UInt64}}
    ## Workspace for data
    dst :: KeyV

    ## Counter; length-1 vector, corresponds to CUB d_ctrs
    tile_counter :: OffsetV

    ## Packed look-back table; flat vector, corresponds to CUB d_lookback
    ##
    ## Each entry uses the highest two bits as state:
    ##   00...0000                  => EMPTY
    ##   01xxxxxx...xxxx            => PARTIAL local count
    ##   10xxxxxx...xxxx            => GLOBAL prefix count
    ##
    ## The lower 62 bits store the count.
    ## Length = 256 * ntile for an 8-bit radix pass.
    lookback :: OffsetV

    ## Bucket offsets; correspond to CUB d_bins_in / d_bins_out.
    ##
    ## Both buffers store 1-based Julia output indices.
    ## bucket_offsets_in[bucket + 1] is the first Julia index
    ## for `bucket` in the current radix pass.
    ## bucket_offsets_out is the output/update buffer for later passes.
    ## Length = 256 for an 8-bit radix pass.
    bucket_offsets_in  :: OffsetV
    bucket_offsets_out :: OffsetV

    function OnesweepWorkspace(:: Type{KeyV}) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}}
        dst = KeyV(undef, 0)

        tile_counter = similar(dst, UInt64, 0)
        lookback = similar(dst, UInt64, 0)
        bucket_offsets_in = similar(dst, UInt64, 0)
        bucket_offsets_out = similar(dst, UInt64, 0)

        OffsetV = typeof(tile_counter)

        return new{KeyT, KeyV, OffsetV}(
            dst,
            tile_counter,
            lookback,
            bucket_offsets_in,
            bucket_offsets_out,
        )
    end
end

function initialize_workspace!(ws :: OnesweepWorkspace{KeyT, KeyV, OffsetV}, nelems :: Int, ntiles :: Int) where {KeyT <: Unsigned, KeyV <: AbstractVector{KeyT}, OffsetV <: AbstractVector{UInt64}}
    # Resize the array
    resize!(ws.dst, nelems)
    resize!(ws.tile_counter, 1)
    resize!(ws.lookback, 256 * ntiles)
    resize!(ws.bucket_offsets_in, 256)
    resize!(ws.bucket_offsets_out, 256)

    ws.tile_counter[1] = UInt64(0)

    fill!(ws.lookback, UInt64(0))
    fill!(ws.bucket_offsets_in, UInt64(0))
    fill!(ws.bucket_offsets_out, UInt64(0))

    return nothing
end