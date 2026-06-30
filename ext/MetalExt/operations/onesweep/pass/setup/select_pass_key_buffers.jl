"""
    _select_pass_key_buffers(codes::KeyV, ws::OnesweepWorkspace, ::Val{Pass})

Return the active Metal key source and destination buffers for one radix pass.

Odd passes read from `codes` and write to `ws.dst`; even passes read from
`ws.dst` and write back to `codes`. This function only returns device-buffer
aliases and does not mutate workspace state.

CUB parallel: the dispatch layer chooses `d_keys_in` and `d_keys_out` before
constructing `AgentRadixSortOnesweep`.

# Parameters

- `codes`: Original Metal key buffer, used as source on odd passes and destination on even passes.
- `ws`: Metal OneSweep workspace containing the ping-pong destination buffer `ws.dst`.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _select_pass_key_buffers end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _select_pass_key_buffers(codes :: KeyV, ws :: OnesweepWorkspace{$KeyT, KeyV, OffsetV}, :: Val{Pass}) where {KeyV <: MtlDeviceVector{$KeyT}, OffsetV <: MtlDeviceVector{UInt32}, Pass}
            if isodd(Pass)
                return codes, ws.dst
            else
                return ws.dst, codes
            end
        end
    end
end
