"""
    radix_sort!(codes, [workspace], [backend parameters...])

Sort unsigned integer keys in place with a backend-specific radix sort.

The current public implementation is provided by the Metal extension for
`MtlVector` inputs. CPU and CUDA inputs use `onesweep_sort!` instead.
"""
function radix_sort! end

"""
    radix_sortperm!(codes, [workspace], [backend parameters...])

Sort unsigned integer keys in place with a backend-specific radix sort and
return stable, 1-based source permutation indices.

The current public implementation is provided by the Metal extension for
`MtlVector` inputs. CPU and CUDA inputs use `onesweep_sortperm!` instead.
"""
function radix_sortperm! end
