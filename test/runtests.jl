# ========================================================================== #
#  UnsignedRadixSorts.jl - Test Suite Entry Point
# ========================================================================== #
#
#  Run with:  julia --project -e "using Pkg; Pkg.test()"
#             or: include("test/runtests.jl") from the REPL
#
#  Ordering convention
#  1. Workspace and per-agent temporary storage
#  2. Process() helper stages: LoadKeys, RankKeys, Lookback, Scatter/Gather
#  3. One-pass Process() dataflow
#  4. Full Onesweep sort / sortperm / workspace reuse
#
# ========================================================================== #

using Test
using UnsignedRadixSorts

include("onesweep_pass_test_utils.jl")
include("onesweep_workspace_tests.jl")
include("onesweep_pass_load_tests.jl")
include("onesweep_pass_rank_tests.jl")
include("onesweep_pass_lookback_tests.jl")
include("onesweep_pass_scatter_tests.jl")
include("onesweep_pass_pipeline_tests.jl")
include("onesweep_sort_tests.jl")
