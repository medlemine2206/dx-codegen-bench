# Pitfalls checklist (read before writing Dx code)

## Compilation

1. **The description is a type.** Build it with `decltype(A() + B() + ...)`; every operator
   is default-constructed with `()`. `Block()` is required for block execution.
2. **Enum spelling:** `cublasdx::type::real` / `cublasdx::type::complex`,
   `cublasdx::function::MM`, `cublasdx::col_major` / `cublasdx::row_major`;
   `cufftdx::fft_type::c2c`, `cufftdx::fft_direction::forward` / `inverse`.
3. **`SM<BENCH_SM>`** — the value is architecture × 10 and must match `-arch`.
4. Traits only exist on a **complete** description (Size + Precision + Type + Function/
   Direction + SM + Block). An incomplete description gives "no member named block_dim".
5. Value types come from the description (`GEMM::c_value_type`, `FFT::value_type`);
   passing `float2*` where `cufftdx::complex<float>*` is expected fails — `reinterpret_cast`.
6. cuBLASDx: `TransposeMode<>` is the legacy way and is deprecated; describe the storage
   order with `Arrangement<>` (row-/column-major) instead.
7. `cublasdx::copy<GEMM, alignment>(src, dst)` needs the alignment as a template argument:
   `cublasdx::alignment_of<GEMM>::a` (`::b`, `::c`).
8. Do not define `main()`; define exactly the requested `solution(...)` function.

## Launch configuration

9. Launch with **exactly** `X::block_dim` threads (a `dim3`, may be 2-D for cuFFTDx).
   Any other block shape is undefined behaviour / assertion failure.
10. Pass the dynamic shared memory size in the launch:
    `cublasdx::get_shared_storage_size<GEMM>()` or `FFT::shared_memory_size`
    (max of both when fusing two descriptions). Above 48 KB, call
    `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.
11. Declare dynamic shared memory aligned: `extern __shared__ __align__(16) char smem[];`.

## Inside the kernel

12. **Never `return` before `execute()`** in a thread of the block (tail blocks!): use an
    `active` flag to mask loads/stores instead. `execute()` contains `__syncthreads()`.
13. cuBLASDx: `copy_wait()` after loading; `__syncthreads()` after `execute()` before
    reading `c_s` or copying it out; another `__syncthreads()` after an epilogue.
14. cuFFTDx: point `n` ↔ thread `n % FFT::stride`, register `n / FFT::stride`; the FFT
    index inside the block is `threadIdx.y`.
15. cuFFTDx inverse transforms are **unnormalised**.
16. `__syncthreads()` between two FFT executions that reuse the same shared memory.
17. Always pass the workspace (`cufftdx::make_workspace<FFT>(err, stream)`) to kernels
    and to `execute()`; it is empty for sizes that do not need one.

## Performance

18. One small FFT or GEMM per block wastes the GPU: pack several per block
    (`FFTsPerBlock`) or give each block enough threads (`BlockDim`).
19. Keep data on chip between fused steps (registers / shared memory); every extra
    kernel launch is a round trip through DRAM.
