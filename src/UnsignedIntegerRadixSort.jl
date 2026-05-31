module UnsignedIntegerRadixSort

using Atomix
using Base.Cartesian
using Base.Threads

# Include source files.
include(joinpath("julia", "struct", "RadixSortWorkspace.jl"))
include(joinpath("julia", "struct", "RadixSortPermWorkspace.jl"))
include(joinpath("julia", "struct", "OnesweepWorkspace.jl"))
include(joinpath("julia", "struct", "global_variable.jl"))
include(joinpath("julia", "operations", "radix_bucket.jl"))
include(joinpath("julia", "operations", "tools.jl"))
include(joinpath("julia", "operations", "radix_histogram.jl"))
include(joinpath("julia", "operations", "radix_offsets.jl"))
include(joinpath("julia", "operations", "radix_scatter.jl"))
include(joinpath("julia", "operations", "radix_pass.jl"))
include(joinpath("julia", "operations", "radix_sort.jl"))
include(joinpath("julia", "operations", "radix_sortperm.jl"))
include(joinpath("julia", "operations", "prepare_bucket_offsets.jl"))
include(joinpath("julia", "operations", "onesweep_pass.jl"))
include(joinpath("julia", "operations", "onesweep_sort.jl"))

# Package metadata helpers.
version() = pkgversion(@__MODULE__)

function about()
    @info "UnsignedIntegerRadixSort Module\n  Version: $(version())\n  Made by Wei-Shan Su, Apr 2026"
    return nothing
end

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
