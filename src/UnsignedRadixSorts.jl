module UnsignedRadixSorts

using Atomix
using Base.Cartesian
using Base.Threads

# Include source files.
include(joinpath("julia", "struct", "OnesweepWorkspace.jl"))
include(joinpath("julia", "struct", "global_variable.jl"))
include(joinpath("julia", "operations", "radix_bucket.jl"))
include(joinpath("julia", "operations", "tools.jl"))
include(joinpath("julia", "operations", "prepare_bucket_offsets.jl"))
include(joinpath("julia", "operations", "onesweep_pass.jl"))
include(joinpath("julia", "operations", "onesweep_sort.jl"))

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
