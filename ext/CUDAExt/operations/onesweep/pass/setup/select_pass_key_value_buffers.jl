"""
    _select_pass_key_value_buffers(codes::KeyV, ws::OnesweepWorkspace, ::Val{Pass})

Return the active CUDA key and permutation source/destination buffers for one
radix pass.

Odd passes read keys from `codes`, write keys to `ws.dst`, read permutation
values from `ws.perms[1]`, and write them to `ws.perms[2]`. Even passes reverse
both ping-pong pairs.

CUB parallel: the agent receives both key and value input/output pointers; here
permutation indices are the values.

# Parameters

- `codes`: Original CUDA key buffer, used as source on odd passes and destination on even passes.
- `ws`: CUDA OneSweep workspace containing key and permutation ping-pong buffers.
- `::Val{Pass}`: Compile-time 1-based radix pass selector.
"""
function _select_pass_key_value_buffers end

for KeyT in (UInt8, UInt16, UInt32, UInt64)
    @eval begin
        @inline function _select_pass_key_value_buffers(codes :: CodeV, ws :: OnesweepWorkspace{$KeyT, WorkspaceKeyV, OffsetV}, :: Val{Pass}) where {CodeV <: CuDeviceVector{$KeyT}, WorkspaceKeyV <: CuDeviceVector{$KeyT}, OffsetV <: CuDeviceVector{UInt32}, Pass}
            # Keep key and permutation ping-pong buffers on the same pass side.
            if isodd(Pass)
                return codes, ws.dst, ws.perms[1], ws.perms[2]
            else
                return ws.dst, codes, ws.perms[2], ws.perms[1]
            end
        end
    end
end
