@inline function _radix_digit(x :: T, N :: Int) where {T <: Unsigned}
    return Base.unsafe_trunc(UInt8, x >> (8 * (N - 1)))
end

@inline function _radix_digit(x :: T, :: Val{N}) where {T <: Unsigned, N}
    return _radix_digit(x, N)
end

@inline function _radix_bucket(x :: T, N :: Int) where {T <: Unsigned}
    return Int(_radix_digit(x, N)) + 1
end

@inline function _radix_bucket(x :: T, :: Val{N}) where {T <: Unsigned, N}
    return _radix_bucket(x, N)
end
