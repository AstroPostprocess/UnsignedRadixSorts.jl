




# Toolbox
@inline function _bucket_offsets_for_pass(ws :: OnesweepWorkspace, :: Val{Pass}) where {Pass}
    if isodd(Pass)
        return ws.bucket_offsets[1], ws.bucket_offsets[2]
    else
        return ws.bucket_offsets[2], ws.bucket_offsets[1]
    end
end