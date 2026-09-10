# Performance

Since `jax` is primarily designed for execution on accelerators (GPUs and TPUs), it is expected that the implemented functions will run faster on these devices for sufficiently large arrays. The plots included in the sections below depict performance measurements regarding [`solve_2d_simple`](docs/solver_2d.md#solve_2d_simple) (and [`solve_2d_fixed_simple`](docs/solver_2d.md#solve_2d_fixed_simple) for `vjp`) on a system with an [AMD Ryzen Threadripper 2950X](https://en.wikichip.org/wiki/amd/ryzen_threadripper/2950x) CPU and a [NVIDIA GeForce RTX 2070S](https://www.techpowerup.com/gpu-specs/geforce-rtx-2070-super.c3440) (or [NVIDIA Titan RTX](https://www.techpowerup.com/gpu-specs/titan-rtx.c3311)) GPU. Note that the theoretical `FP64` performance of these GPUs is very poor compared to a device such as the [NVIDIA A100](https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623). The benchmarks use the JIT-compiled wrapper functions `solve`, `jvp`, and `vjp`, where `solve` maps a primal vector to the computed solution, and `jvp` (`vjp`) maps primal and tangent (cotangent) vectors to a tuple containing the solution and the resulting product (see [benchmark.py](examples/benchmark.py) for details). Since `jax.jit` cannot assume anything (other than type and shape) about the values contained in these vectors, these benchmarks are worst-case scenarios; in practical applications some of the solver parameters can often be constants (i.e., *static* with respect to `jax.jit`). Unless noted otherwise, [`jax-v0.5.0`](https://pypi.org/project/jax/0.5.0/) was used to generate the shown data.

## CPU vs. GPU

The following speedup was observed across problems (test cases) when using an `RTX 2070S` GPU compared to the [CPU-only setting described below](#cpu-only-systems) for a resolution of `128 x 128`, which seems to be the crossover point for `problem 1` (i.e., where the GPU-accelerated program starts to outperform the CPU-only version).

![Speedup gpu_resolution128](data/speedup_gpu_res128.svg)

The speedup will be much larger for higher resolutions. To illustrate this, the following plot shows how the speedup scales with resolution in the case of `solve` calls for `problem 1`.

![Speedup_vs_resolution_problem1](data/p1_solve_speedup.svg)

Qualitatively, the graph looks about the same for `jvp`, and similar for `vjp` calls. However, note that you may run into [memory issues on GPU](#memory-allocation-gpu), especially for `vjp`.

<a id="cpu-only-systems"></a>

## CPU-Only Systems

In [`jaxlib-v0.4.32`](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md#jaxlib-0432-september-11-2024) a *new version of the CPU backend* was implemented, which (while improving compilation times) is **extremely detrimental** to the runtime performance of this solver.

If maximum performance on a CPU-only system is required:

- try downgrading to an earlier version such as [`jax-v0.4.31`](https://pypi.org/project/jax/0.4.31/), which supports Python `3.10–3.12`, or
- for affected versions with `jax < v0.6.0` (e.g., [`jax-v0.5.0`](https://pypi.org/project/jax/0.5.0/), which supports Python `3.10–3.13`), switch to the old CPU backend by setting the environment variable `XLA_FLAGS='--xla_cpu_use_thunk_runtime=false'`.

The following plot shows the speedup of setting this flag to `false` when using a resolution of `128 x 128`.

![Speedup when xla_cpu_use_thunk_runtime is set to false resolution 128](data/speedup_xlaflag_res128.svg)

In higher-resolutions settings this seems to become more uniform with a speedup of about `4` for all measured problems. Other plots in this document refer to the setup where the flag is set to `false` as `cpu_xlaflag`.

<a id="memory-allocation-gpu"></a>

## Memory Allocation (GPU)

For higher resolutions you might encounter some form of "out of memory" (OOM) error, which may be avoidable by choosing a different memory allocation strategy. The page [gpu memory allocation](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) shows environment variables that can be used to control the memory allocation behavior. Setting `XLA_PYTHON_CLIENT_ALLOCATOR=platform` is very useful if executing a function slightly exceeds the available device memory; however, it does reduce performance a bit (e.g., by a few percent when solving `problem 1`).

## Memory Consumption

A lot of memory is consumed with higher resolutions.

Since peak memory consumption depends on the exact behavior of the underlying memory management, actual measurements might not yield a *tight* upper bound on the memory requirement. Reasonable lower bounds were obtained by using [AOT compilation](https://jax.readthedocs.io/en/latest/aot.html) and [memory analysis](https://jax.readthedocs.io/en/latest/jax.stages.html#jax.stages.Compiled.memory_analysis) to obtain estimated memory requirements. Note that they are still very platform dependent. The following plots illustrate these results for `problem 1`. It is important to note that the labels on the `x-axis` change for `vjp` since it requires significantly more memory than `jvp`.

In the CPU-only setting, the *maximum resident set size* is measured by using [`time -v`](https://man7.org/linux/man-pages/man1/time.1.html) when running [benchmark.py](examples/benchmark.py).

![Memory requirements problem 1 solve CPU](data/p1_solve_cpu_xlaflag_mem_req.svg)

![Memory requirements problem 1 jvp CPU](data/p1_jvp_cpu_xlaflag_mem_req.svg)

![Memory requirements problem 1 vjp CPU](data/p1_vjp_cpu_xlaflag_mem_req.svg)

For GPUs (here: using the `RTX 2070S`), `XLA_PYTHON_CLIENT_ALLOCATOR` is set to `platform` in order to actually deallocate unused memory, and a monitoring [thread](https://docs.python.org/3/library/threading.html) uses [`pynvml`](https://pypi.org/project/pynvml/) to sample the amount of used memory during the respective function call.

![Memory requirements problem 1 solve GPU (2070S)](data/p1_solve_gpu_mem_req.svg)

![Memory requirements problem 1 jvp GPU (2070S)](data/p1_jvp_gpu_mem_req.svg)

![Memory requirements problem 1 vjp GPU (2070S)](data/p1_vjp_gpu_mem_req.svg)

## Compilation Time

The [persistent compilation cache](https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html) can be used to cache compiled functions on the local disk. This avoids recompilation across script runs, but it still takes a bit of time for `jax` to manage this cache.

```python
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
```

Note that, unless the solver performs a very large number of iterations or the resolution is very high, the warm-up time (mainly compilation) is extremely high compared to the execution time of a *repeated* run since the multigrid setup introduces many instructions (different array shapes on each level).

The following plots illustrate the ratio between warm-up run and repeated runs:

![Warmup ratio CPU](data/p1_cpu_xlaflag_warmup_ratio.svg)

![Warmup ratio GPU (2070S)](data/p1_gpu_warmup_ratio.svg)

