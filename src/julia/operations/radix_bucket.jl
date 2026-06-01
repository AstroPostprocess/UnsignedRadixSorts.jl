"""
    _radix_digit(x::T, N::Int) where {T <: Unsigned}

Return the 8-bit radix digit selected by the 1-based pass index.

# Parameters

- `x`: Unsigned integer value to inspect.
- `N`: 1-based radix pass index.

# Returns

The selected digit as a `UInt8`.
"""
@inline function _radix_digit(x :: T, N :: Int) where {T <: Unsigned}
    return Base.unsafe_trunc(UInt8, x >> (8 * (N - 1)))
end

"""
    _radix_bucket(x::T, N::Int) where {T <: Unsigned}

Return the 1-based radix bucket selected by the pass index.

# Parameters

- `x`: Unsigned integer value to inspect.
- `N`: 1-based radix pass index.

# Returns

The selected bucket index as an `Int`.
"""
@inline function _radix_bucket(x :: T, N :: Int) where {T <: Unsigned}
    return Int(_radix_digit(x, N)) + 1
end
