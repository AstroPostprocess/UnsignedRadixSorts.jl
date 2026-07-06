# UnsignedRadixSorts.jl

`UnsignedRadixSorts.jl` is an experimental radix sorting package for unsigned integer keys in Julia. It implements an 8-bit LSD [OneSweep](https://arxiv.org/abs/2206.01784)-style radix sorter for in-place key sorting and permutation sorting, with CPU, CUDA, and Metal backends. The package is intended as a lightweight sorting primitive for workflows where unsigned integer keys, such as Morton codes, are already available in memory and need to be reordered without leaving the active backend.

The implementation follows the high-level structure of CUB/CCCL’s OneSweep radix sort: each radix pass computes tile-local ranks, publishes per-tile counts, resolves global offsets through decoupled lookback, and scatters keys to the output side of a ping-pong buffer. The CPU implementation is threaded. CUDA and Metal support are provided through Julia package extensions.

## Status

`UnsignedRadixSorts.jl` is currently experimental. The CPU backend is the primary implementation. CUDA and Metal support are provided for backend-resident workflows where avoiding host-device round trips may be more important than matching specialised vendor-style sorting libraries.

For users primarily looking for maximum CUDA sorting performance, [`KernelForge.jl`](https://github.com/epilliat/KernelForge.jl) is currently the more appropriate choice. It is faster and more mature for CUDA workloads. `UnsignedRadixSorts.jl` is not intended to replace mature CUDA-focused sorting libraries; its goal is to provide a small unsigned-integer sorting primitive with a consistent API across CPU, CUDA, and Metal.

The Metal backend uses a Metal-safe reread-scatter staging path instead of the CUDA-style shared keys_out[rank] staging path. This is intended to preserve correctness on Apple GPUs, but it is not expected to match CUDA/CUB-level performance, especially for small arrays where Metal launch and pass overhead dominate.

The public API and backend performance characteristics may still change before a stable 1.0 release.



## Usage

The package exposes two main entry points:

- `onesweep_sort!(codes)` for in-place sorting of unsigned integer keys
- `onesweep_sortperm!(codes)` for in-place sorting of keys while returning the corresponding 1-based source permutation indices

Supported key types are unsigned integer vectors. The CPU implementation supports `UInt8`, `UInt16`, `UInt32`, `UInt64`, and `UInt128`. The CUDA and Metal extensions currently target `UInt8`, `UInt16`, `UInt32`, and `UInt64` device vectors.

Install from a local checkout or repository URL:

```julia
using Pkg
Pkg.develop(path = "/path/to/UnsignedRadixSorts.jl")
# or
Pkg.add(url = "https://github.com/AstroPostprocess/UnsignedRadixSorts.jl")
```

After installation, a typical CPU usage looks like:

```julia
using UnsignedRadixSorts
using Random

Random.seed!(1234)
keys = rand(UInt64, 1_000_000)

# Sort in-place; keys is modified.
onesweep_sort!(keys)
issorted(keys)
```

To obtain the source permutation indices, use `onesweep_sortperm!`:

```julia
using UnsignedRadixSorts
using Random

Random.seed!(1234)
original = rand(UInt32, 1_000_000)
keys = copy(original)

perm = onesweep_sortperm!(keys)

issorted(keys)
original[Int.(perm)] == keys
```

`onesweep_sortperm!` sorts `keys` in-place and returns a permutation buffer. The permutation entries are 1-based indices into the original key order.

For repeated sorts, a workspace can be created and reused explicitly:

```julia
using UnsignedRadixSorts

keys = rand(UInt32, 1_000_000)
ws = OnesweepWorkspace(Vector{UInt32})

onesweep_sort!(keys, ws)
```

The default CPU tile size is `4096`, but it can be supplied as a compile-time value:

```julia
onesweep_sort!(keys, Val(4096))
onesweep_sortperm!(keys, Val(4096))
```



## CUDA and Metal backends

CUDA support is loaded through `CUDAExt` when `CUDA.jl` is available:

```julia
using UnsignedRadixSorts
using CUDA

keys = CuVector(rand(UInt64, 1_000_000))

onesweep_sort!(keys)
CUDA.synchronize()
```

The CUDA entry points also accept backend parameters:

```julia
onesweep_sort!(keys, Val(4096), Val(256), Val(256))
```

where the arguments are tile size, number of CUDA blocks, and threads per block.

Metal support is loaded through `MetalExt` when `Metal.jl` is available:

```julia
using UnsignedRadixSorts
using Metal

keys = MtlVector(rand(UInt64, 1_000_000))

onesweep_sort!(keys)
Metal.synchronize()
```

The Metal entry points use a smaller default tile size and expose the number of threadgroups and threads per threadgroup:

```julia
onesweep_sort!(keys, Val(2048), Val(128), Val(256))
```

The Metal backend is experimental. It is mainly intended for GPU-resident workflows where keys are already on Apple GPU memory and the sorted result is consumed by later GPU kernels. It should not be interpreted as a drop-in replacement for highly tuned CPU sorting on small arrays.



## Algorithm outline

`UnsignedRadixSorts.jl` implements a stable 8-bit LSD radix sort over unsigned integer keys. A key type with `B` bytes is sorted using `B` radix passes:

1. `UInt8`: 1 pass
2. `UInt16`: 2 passes
3. `UInt32`: 4 passes
4. `UInt64`: 8 passes
5. `UInt128`: 16 passes on the CPU backend

Each pass uses 256 buckets. The high-level pass structure is:

1. **Bucket-offset preparation**: compute global bucket starts for each byte pass.
2. **Tile claiming**: workers or GPU threadgroups dynamically claim tiles of the input.
3. **Key loading**: load the current tile from the active ping-pong source buffer.
4. **Local ranking**: compute stable tile-local ranks within each radix bucket.
5. **Partial publication**: publish per-tile bucket counts into a packed lookback table.
6. **Decoupled lookback**: scan previous tiles' same-bucket counts to obtain each tile's global prefix.
7. **Global scatter**: write keys to the active destination buffer using the bucket start, previous-tile prefix, and local rank.
8. **Ping-pong pass iteration**: alternate between the original key buffer and workspace destination buffer for each radix byte.

The core output-index relation is:

```text
output_index = bucket_start + previous_tile_count + local_rank
```

where `bucket_start` is the global start of the radix bucket, `previous_tile_count` is the number of keys in the same bucket from earlier tiles, and `local_rank` is the stable rank of the key within the current tile.

The sort is stable at the radix-pass level. `onesweep_sortperm!` uses the same key scatter positions to move 1-based source indices alongside the keys.



## Implementation notes

The package contains three related implementations:

1. CPU backend: threaded Julia implementation using reusable workspace buffers and Atomix counters.
2. CUDA backend: CUDA extension following a CUB-like OneSweep staging path using shared-memory key exchange.
3. Metal backend: Metal extension following the same OneSweep lookback structure, but with a Metal-specific scatter staging path.

The CUDA backend follows the CUB/CCCL-style tile flow closely: keys and ranks are kept in thread-private storage, keys are staged through a shared `keys_out[rank]` permutation buffer, and global scatter reads the locally reordered tile.

The Metal backend intentionally differs at this point. The CUDA-style path

```text
MVector ranks -> keys_out[rank] = key -> global scatter
```

has been unstable on Metal. The Metal implementation therefore stages local ranks in threadgroup memory and rereads keys from the source buffer during global scatter. This adds a sequential source read, but keeps the decoupled-lookback OneSweep invariant intact and has been the more reliable path for large GPU-resident sorts on Apple GPUs.

Compared with a full production sorting library, this package intentionally keeps the scope narrow:

1. Unsigned integer keys only.
2. 8-bit radix digits only.
3. In-place key sorting and key-associated permutation sorting only.
4. No comparison-based fallback path inside the package API.
5. No signed integer, floating-point, custom-order, or key-value payload API beyond source-index permutation sorting.
6. Metal support is correctness-oriented and experimental, especially for small arrays where fixed launch and pass overheads dominate.

