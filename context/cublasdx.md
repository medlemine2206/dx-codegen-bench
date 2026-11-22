# cuBLASDx 0.4 — GEMM inside your kernel

## 1. Description

```cpp
#include <cublasdx.hpp>

using GEMM = decltype(cublasdx::Size<M, N, K>()                 // C is M x N, A is M x K, B is K x N
                    + cublasdx::Precision<float>()              // or <__half, __half, float>, <double>, ...
                    + cublasdx::Type<cublasdx::type::real>()    // or cublasdx::type::complex
                    + cublasdx::Function<cublasdx::function::MM>()
                    + cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>()
                    + cublasdx::Block()                         // block-wide execution
                    + cublasdx::BlockDim<128>()                 // optional; default = suggested
                    + cublasdx::SM<BENCH_SM>());                // must match -arch
```

* `Precision<TA, TB, TC>`: input types of A and B and the type of C / accumulation.
  `Precision<__half, __half, float>` = half inputs, **float accumulation and float C**.
* `Type<complex>`: the value type becomes `cublasdx::complex<float>` (layout-compatible
  with `float2` / `cuComplex`; members `.x`, `.y`, or `.real()`, `.imag()`). Cast device
  pointers with `reinterpret_cast`. Alpha/beta are then complex: `value_type{1.f, 0.f}`.
* `Arrangement<A, B, C>` = storage order of each matrix: `cublasdx::col_major` or
  `cublasdx::row_major` (defaults: col-major). The legacy `TransposeMode<>` operator is
  deprecated — describe how the data is laid out instead.
* Value types: `GEMM::a_value_type`, `GEMM::b_value_type`, `GEMM::c_value_type`.
* Launch info: `GEMM::block_dim` (dim3), `GEMM::max_threads_per_block`,
  `cublasdx::get_shared_storage_size<GEMM>()` (bytes of dynamic shared memory for A, B, C).

## 2. Canonical kernel (one block computes one GEMM)

```cuda compile-check
#include <cublasdx.hpp>

constexpr int M = 16, N = 32, K = 8;
using GEMM = decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
                      cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<BENCH_SM>());

__global__ void __launch_bounds__(GEMM::max_threads_per_block)
gemm_kernel(const float* A, const float* B, float* C, float alpha, float beta) {
    extern __shared__ __align__(16) char smem[];

    // global-memory views (layout follows the Arrangement operator)
    auto a_g = cublasdx::make_tensor(A, GEMM::get_layout_gmem_a());
    auto b_g = cublasdx::make_tensor(B, GEMM::get_layout_gmem_b());
    auto c_g = cublasdx::make_tensor(C, GEMM::get_layout_gmem_c());

    // carve the dynamic shared memory into A, B, C (correctly aligned)
    auto [sa, sb, sc] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a_s = cublasdx::make_tensor(sa, GEMM::get_layout_smem_a());
    auto b_s = cublasdx::make_tensor(sb, GEMM::get_layout_smem_b());
    auto c_s = cublasdx::make_tensor(sc, GEMM::get_layout_smem_c());

    using align = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, align::a>(a_g, a_s);   // cooperative (all threads), asynchronous
    cublasdx::copy<GEMM, align::b>(b_g, b_s);
    cublasdx::copy<GEMM, align::c>(c_g, c_s);   // only needed when beta != 0
    cublasdx::copy_wait();                      // waits for the copies AND does __syncthreads()

    GEMM().execute(alpha, a_s, b_s, beta, c_s); // C = alpha*A*B + beta*C, in shared memory
    __syncthreads();                            // results visible to every thread

    cublasdx::copy<GEMM, align::c>(c_s, c_g);   // shared -> global
}

void launch(const float* A, const float* B, float* C, cudaStream_t stream) {
    gemm_kernel<<<1, GEMM::block_dim, cublasdx::get_shared_storage_size<GEMM>(), stream>>>(A, B, C, 1.f, 0.f);
}
```

## 3. Batching: one block per matrix

For `batch` independent matrices stored back to back, launch `batch` blocks and offset the
pointers by `blockIdx.x * (rows * cols)` before building the global tensors.
`get_layout_gmem_*()` describes a single, densely packed matrix.

## 4. Fusing work around the GEMM (epilogue / prologue)

After `execute()` + `__syncthreads()`, `c_s` lives in shared memory and can be read and
written by any thread with 2-D indexing: `c_s(i, j)` (row i, column j, whatever the
arrangement). Typical epilogue:

```cpp
for (int e = threadIdx.x; e < M * N; e += blockDim.x) {
    const int i = e % M, j = e / M;
    c_s(i, j) = some_function(c_s(i, j), i, j);
}
__syncthreads();                    // before the copy back to global memory
cublasdx::copy<GEMM, align::c>(c_s, c_g);
```
A prologue works the same way on `a_s` / `b_s` after `copy_wait()`.

## 5. Raw-pointer overload (no tensors)

`GEMM().execute(alpha, sa_ptr, lda, sb_ptr, ldb, beta, sc_ptr, ldc)` with shared-memory
pointers and leading dimensions (for column-major: ld = number of rows). Useful when the
operands are sub-blocks of a bigger shared buffer.

## 6. Shared memory above 48 KB

If `get_shared_storage_size<GEMM>()` exceeds 48 KB, opt in before launching:
`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);`
