# ========================================================================== #
#  UnsignedIntegerRadixSort.jl - Test Suite Entry Point
# ========================================================================== #
#
#  Run with:  julia --project -e "using Pkg; Pkg.test()"
#             or: include("test/runtests.jl") from the REPL
#
#  Ordering convention
#  1. Bucket-offset preparation
#  2. Full Onesweep sort
#  3. Onesweep sortperm / stability
#  4. Workspace reuse
#
# ========================================================================== #

using Test
using UnsignedIntegerRadixSort

include("onesweep_sort_tests.jl")
