# MathDx device extensions — context pack for code generation

You are writing CUDA C++ that calls NVIDIA **device-side** math libraries ("Dx"):
the library functions are called **from inside your own `__global__` kernel**, by all
threads of a block, on data in registers / shared memory. Nothing is launched by the
library itself: you write the kernel, you choose the launch configuration that the
library description tells you.

Versions this pack was written for (MathDx 25.06): **cuBLASDx 0.4.0**, **cuFFTDx 1.5.0**,
cuSolverDx 0.2.0, cuRANDDx 0.2.0. Generated API indexes in `generated/` list every
operator, enum value and trait of the installed headers.

## The one pattern behind every Dx library

1. **Describe** the operation at compile time by adding operators:
   `using X = decltype(lib::Size<...>() + lib::Precision<...>() + ... + lib::Block());`
2. **Read the launch configuration from the description**: `X::block_dim`,
   shared memory size (`X::shared_memory_size` for cuFFTDx,
   `cublasdx::get_shared_storage_size<X>()` for cuBLASDx).
3. **Launch** your kernel with exactly that block shape and dynamic shared memory.
4. **Inside the kernel**, every thread of the block calls `X().execute(...)`.
   Never return early before `execute()`: it contains block-wide barriers.

## Build line (fixed by the harness, you do not control it)

```
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_XY -DBENCH_SM=XY0 \
     -I<mathdx>/include -I<mathdx>/external/cutlass/include  harness.cu
```

* `BENCH_SM` is the target architecture × 10 (T4 → 750, A100 → 800, L4 → 890).
  Use it in the `SM<BENCH_SM>` operator: the SM operator **must** match the compiled
  architecture.
* Include `<cublasdx.hpp>` and/or `<cufftdx.hpp>`; `cuda_runtime.h`, `cuda_fp16.h` are
  already included by the harness.

## Files in this pack

* `cublasdx.md` — GEMM inside a kernel: description, tensors, copies, execute, epilogues.
* `cufftdx.md` — FFT inside a kernel: description, data distribution, workspace, fusion.
* `pitfalls.md` — checklist of the mistakes that break compilation or results.
* `generated/*_api.md` — exhaustive API index extracted from the installed headers.
