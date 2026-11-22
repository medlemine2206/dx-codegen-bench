# cuFFTDx 1.5 — FFT inside your kernel

## 1. Description

```cpp
#include <cufftdx.hpp>

using FFT = decltype(cufftdx::Size<128>()
                   + cufftdx::Precision<float>()                       // float or double (or __half)
                   + cufftdx::Type<cufftdx::fft_type::c2c>()           // c2c, r2c, c2r
                   + cufftdx::Direction<cufftdx::fft_direction::forward>()   // or ::inverse
                   + cufftdx::ElementsPerThread<8>()                   // optional (suggested otherwise)
                   + cufftdx::FFTsPerBlock<2>()                        // optional, default 1
                   + cufftdx::SM<BENCH_SM>()                           // must match -arch
                   + cufftdx::Block());
using complex_type = typename FFT::value_type;   // cufftdx::complex<float>: .x (re), .y (im)
```

Traits used for the launch and the data distribution:

| trait | meaning |
|---|---|
| `FFT::block_dim` | dim3 to launch with: `x` threads per FFT, `y = ffts_per_block` |
| `FFT::shared_memory_size` | bytes of dynamic shared memory for the whole block |
| `FFT::elements_per_thread` | how many points of one FFT each thread holds |
| `FFT::storage_size` | size of the per-thread register array (>= elements_per_thread) |
| `FFT::stride` | distance between two points held by the same thread (= `block_dim.x`) |
| `FFT::ffts_per_block` | number of FFTs computed by one block |
| `FFT::requires_workspace`, `FFT::workspace_type` | workspace (needed by some sizes) |
| `cufftdx::size_of<FFT>::value` | the FFT size N |

* `Direction<inverse>` is **not normalised**: divide by N yourself if needed.
* `cufftdx::complex<float>` is layout-compatible with `float2`; `complex<double>` with
  `double2`. Use `reinterpret_cast` on device pointers.

## 2. Data distribution (c2c, natural order)

Point `n` of an FFT is held by thread `threadIdx.x = n % stride`, at register index
`i = n / stride`, i.e. **thread `t`, element `i` ↔ index `t + i * FFT::stride`**.
The output of a c2c transform uses the same mapping (natural order).
`threadIdx.y` selects which FFT of the block the thread works on.

## 3. Canonical kernel

```cuda compile-check
#include <cufftdx.hpp>

constexpr unsigned NFFT = 128;
using FFT = decltype(cufftdx::Size<NFFT>() + cufftdx::Precision<float>() +
                     cufftdx::Type<cufftdx::fft_type::c2c>() +
                     cufftdx::Direction<cufftdx::fft_direction::forward>() +
                     cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<2>() +
                     cufftdx::SM<BENCH_SM>() + cufftdx::Block());
using complex_type = typename FFT::value_type;

__global__ void __launch_bounds__(FFT::max_threads_per_block)
fft_kernel(complex_type* data, int batch, typename FFT::workspace_type workspace) {
    extern __shared__ __align__(16) unsigned char smem[];
    complex_type thread_data[FFT::storage_size];

    const unsigned fft_id = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const bool active = fft_id < (unsigned)batch;      // tail block: do NOT return early
    complex_type* signal = data + (size_t)fft_id * NFFT;

    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned n = threadIdx.x + i * FFT::stride;
        thread_data[i] = (active && n < NFFT) ? signal[n] : complex_type{0.f, 0.f};
    }

    FFT().execute(thread_data, smem, workspace);      // all threads of the block, always

    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned n = threadIdx.x + i * FFT::stride;
        if (active && n < NFFT) signal[n] = thread_data[i];
    }
}

void launch(complex_type* data, int batch, cudaStream_t stream) {
    cudaError_t err;
    auto workspace = cufftdx::make_workspace<FFT>(err, stream);   // cheap/empty when not required
    const unsigned blocks = (batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
    fft_kernel<<<blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(data, batch, workspace);
}
```

## 4. Choosing ElementsPerThread / FFTsPerBlock

* Threads per FFT = `N / elements_per_thread` (e.g. N = 64 with 8 per thread → 8 threads).
* Small FFTs: increase `FFTsPerBlock` so a block has ≥ 64–128 threads; one tiny FFT per
  block leaves the GPU mostly idle.
* Large FFTs: 8–16 elements per thread are typical. Omit `ElementsPerThread` to get
  `FFT::suggested_elements_per_thread`.

## 5. Fusion: several transforms / pointwise math in one kernel

Data stays in `thread_data` between transforms, so FFT → pointwise op → inverse FFT is a
single kernel:

```cpp
using FFT_base = decltype(cufftdx::Size<N>() + cufftdx::Precision<float>() + cufftdx::Type<cufftdx::fft_type::c2c>()
                        + cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<BENCH_SM>() + cufftdx::Block());
using FFT  = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());
// launch with FFT::block_dim and max(FFT::shared_memory_size, IFFT::shared_memory_size)
FFT().execute(thread_data, smem, ws_fwd);
// point i of this thread is index threadIdx.x + i * FFT::stride: multiply, scale, ...
__syncthreads();                    // shared memory is reused by the next transform
IFFT().execute(thread_data, smem, ws_inv);
```
Forward and inverse descriptions must share size, precision, elements per thread and
FFTs per block, so that they have the same `block_dim` and the same data distribution.

## 6. Double precision

`Precision<double>`, value type `cufftdx::complex<double>` (= `double2` layout). Double
FFTs need more shared memory and registers per point; keep `ElementsPerThread` moderate.
