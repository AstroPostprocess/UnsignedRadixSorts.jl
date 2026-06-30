module UnsignedRadixSorts

using Adapt
using Atomix
using Base.Cartesian
using Base.Threads

# Include source files.
include(joinpath("julia", "struct", "OnesweepWorkspace.jl"))
include(joinpath("julia", "struct", "global_variable.jl"))
include(joinpath("julia", "operations", "common", "radix_bucket.jl"))
include(joinpath("julia", "operations", "common", "tools.jl"))
include(joinpath("julia", "operations", "histogram", "prepare_bucket_offsets.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "setup", "select_pass_key_buffers.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "setup", "select_pass_key_value_buffers.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "setup", "claim_next_tile.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "setup", "clear_tile_storage.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "rank", "load_keys_and_count_digits.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "rank", "rank_keys_local.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "lookback", "publish_lookback_partial.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "lookback", "resolve_lookback_global_offsets.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "scatter", "scatter_keys_global.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "scatter", "scatter_key_values_global.jl"))
include(joinpath("julia", "operations", "onesweep", "pass", "onesweep_pass.jl"))
include(joinpath("julia", "operations", "onesweep", "sort", "onesweep_sort.jl"))

# Package metadata helpers.
version() = pkgversion(@__MODULE__)

function about()
    @info "UnsignedRadixSorts Module\n  Version: $(version())\n  Made by Wei-Shan Su, June 2026"
    return nothing
end

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
