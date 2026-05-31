# ========================================================================== #
#  UnsignedIntegerRadixSort.jl - Test Suite Entry Point
# ========================================================================== #
#
#  Run with:  julia --project -e "using Pkg; Pkg.test()"
#             or: include("test/runtests.jl") from the REPL
#
#  Ordering convention
#  1. Radix helpers                  (digit extraction, bucket indexing)
#  2. Histogram / offsets / scatter  (core counting-sort building blocks)
#  3. Single-pass behavior           (stable 8-bit radix passes)
#  4. Full sort                      (public in-place radix_sort!)
#  5. Sortperm / stability           (public radix_sortperm!)
#
# ========================================================================== #

using Test
using UnsignedIntegerRadixSort

include("radix_sort_tests.jl")
include("onesweep_sort_tests.jl")
