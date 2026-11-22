// Reference: inverse FFT (cuFFTDx does not normalise) followed by a 1/N scale on registers.
#include <cufftdx.hpp>

namespace f2 {
constexpr unsigned NFFT = 1024;
using FFT = decltype(cufftdx::Size<NFFT>() + cufftdx::Precision<float>() +
                     cufftdx::Type<cufftdx::fft_type::c2c>() +
                     cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                     cufftdx::ElementsPerThread<16>() + cufftdx::FFTsPerBlock<1>() +
                     cufftdx::SM<BENCH_SM>() + cufftdx::Block());
using complex_type = typename FFT::value_type;

__global__ void __launch_bounds__(FFT::max_threads_per_block)
kernel(const float2* in, float2* out, int batch, typename FFT::workspace_type workspace) {
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    complex_type thread_data[FFT::storage_size];
    const unsigned fft_id = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const bool active = fft_id < (unsigned)batch;
    const complex_type* src = reinterpret_cast<const complex_type*>(in) + (size_t)fft_id * NFFT;
    complex_type* dst = reinterpret_cast<complex_type*>(out) + (size_t)fft_id * NFFT;

#pragma unroll
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned idx = threadIdx.x + i * FFT::stride;
        thread_data[i] = (active && idx < NFFT) ? src[idx] : complex_type{0, 0};
    }
    FFT().execute(thread_data, smem, workspace);
    const float scale = 1.f / NFFT;
#pragma unroll
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned idx = threadIdx.x + i * FFT::stride;
        if (active && idx < NFFT) dst[idx] = complex_type{thread_data[i].x * scale, thread_data[i].y * scale};
    }
}
}  // namespace f2

void solution(const float2* in, float2* out, int batch, cudaStream_t stream) {
    using namespace f2;
    cudaError_t err;
    auto workspace = cufftdx::make_workspace<FFT>(err, stream);
    const unsigned blocks = (batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
    kernel<<<blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(in, out, batch, workspace);
}
