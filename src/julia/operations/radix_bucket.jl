@inline function _radix_digit(x :: T, :: Val{N}) where {T <: Unsigned, N}
    return Base.unsafe_trunc(UInt8, x >> (8 * (N - 1)))
end

@inline function _radix_bucket(x :: T, :: Val{N}) where {T <: Unsigned, N}
    return Int(_radix_digit(x, Val(N))) + 1
end
