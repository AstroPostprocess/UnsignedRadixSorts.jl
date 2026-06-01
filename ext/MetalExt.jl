module MetalExt

using Base.Cartesian
using Base.Threads
using Atomix
using Metal
using Reexport
using UnsignedRadixSorts

include(joinpath(@__DIR__, "MetalExt", "struct", "OnesweepWorkspace.jl"))
include(joinpath(@__DIR__, "MetalExt", "operations", "prepare_bucket_offsets.jl"))
include(joinpath(@__DIR__, "MetalExt", "operations", "onesweep_pass.jl"))
include(joinpath(@__DIR__, "MetalExt", "operations", "onesweep_sort.jl"))


end
