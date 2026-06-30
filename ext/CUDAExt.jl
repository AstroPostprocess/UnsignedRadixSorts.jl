module CUDAExt

using Base.Cartesian
using CUDA
using Reexport
using UnsignedRadixSorts

include(joinpath(@__DIR__, "CUDAExt", "struct", "OnesweepWorkspace.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "histogram", "prepare_bucket_offsets.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "setup", "select_pass_key_buffers.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "setup", "select_pass_key_value_buffers.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "setup", "claim_next_tile.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "setup", "clear_tile_storage.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "rank", "load_keys_and_count_digits.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "rank", "rank_keys_local.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "lookback", "publish_lookback_partial.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "lookback", "resolve_lookback_global_offsets.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "scatter", "scatter_keys_global.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "scatter", "scatter_key_values_global.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "pass", "onesweep_pass.jl"))
include(joinpath(@__DIR__, "CUDAExt", "operations", "onesweep", "sort", "onesweep_sort.jl"))

end
