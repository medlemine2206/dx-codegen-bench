// Reference: forward FFT -> pointwise multiply by the filter spectrum -> inverse FFT, all
// in ONE kernel; data stays in registers between the two transforms.
#include <cufftdx.hpp>

namespace f4 {
constexpr unsigned NFFT = 512;
using FFT_base = decltype(cufftdx::Size<NFFT>() + cufftdx::Precision<float>() +
                          cufftdx::Type<cufftdx::fft_type::c2c>() +
                          cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<2>() +
                          cufftdx::SM<BENCH_SM>() + cufftdx::Block());
using FFT  = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());
using complex_type = typename FFT::value_type;
static_assert(FFT::block_dim.x == IFFT::block_dim.x && FFT::block_dim.y == IFFT::block_dim.y,
              "forward and inverse must share the block shape");
constexpr unsigned smem_bytes =
    FFT::shared_memory_size > IFFT::shared_memory_size ? FFT::shared_memory_size : IFFT::shared_memory_size;

__global__ void __launch_bounds__(FFT::max_threads_per_block)
kernel(const float2* x, const float2* H, float2* y, int batch, typename FFT::workspace_type ws_f,
       typename IFFT::workspace_type ws_i) {
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    complex_type thread_data[FFT::storage_size];
    const unsigned fft_id = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const bool active = fft_id < (unsigned)batch;
    const complex_type* src = reinterpret_cast<const complex_type*>(x) + (size_t)fft_id * NFFT;
    complex_type* dst = reinterpret_cast<complex_type*>(y) + (size_t)fft_id * NFFT;

#pragma unroll
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned idx = threadIdx.x + i * FFT::stride;
        thread_data[i] = (active && idx < NFFT) ? src[idx] : complex_type{0, 0};
    }
    FFT().execute(thread_data, smem, ws_f);
    const float scale = 1.f / NFFT;
#pragma unroll
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {  // same index map before/after
        const unsigned idx = threadIdx.x + i * FFT::stride;
        const float2 h = (idx < NFFT) ? H[idx] : make_float2(0.f, 0.f);
        const complex_type v = thread_data[i];
        thread_data[i] = complex_type{(v.x * h.x - v.y * h.y) * scale, (v.x * h.y + v.y * h.x) * scale};
    }
    __syncthreads();  // shared memory is reused by the second transform
    IFFT().execute(thread_data, smem, ws_i);
#pragma unroll
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned idx = threadIdx.x + i * FFT::stride;
        if (active && idx < NFFT) dst[idx] = thread_data[i];
    }
}
}  // namespace f4

void solution(const float2* x, const float2* H, float2* y, int batch, cudaStream_t stream) {
    using namespace f4;
    cudaError_t err;
    auto ws_f = cufftdx::make_workspace<FFT>(err, stream);
    auto ws_i = cufftdx::make_workspace<IFFT>(err, stream);
    const unsigned blocks = (batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
    kernel<<<blocks, FFT::block_dim, smem_bytes, stream>>>(x, H, y, batch, ws_f, ws_i);
}
