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

    ## bucket_offsets_out is the output/update buffer for bucket offsets,
    ## corresponding to CUB d_bins_out.
    ## Stores 1-based Julia output indices.
    ## bucket_offsets_in[bucket + 1] is the first Julia index
    ## for `bucket` in the current radix pass.
    ## Length = 256 for an 8-bit radix pass.
    bucket_offsets_in  :: OffsetV
    bucket_offsets_out :: OffsetV
end

