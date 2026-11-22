// Reference: double-precision FFT with a fused |X|^2 epilogue.
#include <cufftdx.hpp>

namespace f5 {
constexpr unsigned NFFT = 128;
using FFT = decltype(cufftdx::Size<NFFT>() + cufftdx::Precision<double>() +
                     cufftdx::Type<cufftdx::fft_type::c2c>() +
                     cufftdx::Direction<cufftdx::fft_direction::forward>() +
                     cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<4>() +
                     cufftdx::SM<BENCH_SM>() + cufftdx::Block());
using complex_type = typename FFT::value_type;

__global__ void __launch_bounds__(FFT::max_threads_per_block)
kernel(const double2* in, double* power, int batch, typename FFT::workspace_type workspace) {
    extern __shared__ __align__(16) unsigned char smem[];
    complex_type thread_data[FFT::storage_size];
    const unsigned fft_id = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const bool active = fft_id < (unsigned)batch;
    const complex_type* src = reinterpret_cast<const complex_type*>(in) + (size_t)fft_id * NFFT;
    double* dst = power + (size_t)fft_id * NFFT;

#pragma unroll
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned idx = threadIdx.x + i * FFT::stride;
        thread_data[i] = (active && idx < NFFT) ? src[idx] : complex_type{0, 0};
    }
    FFT().execute(thread_data, smem, workspace);
#pragma unroll
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        const unsigned idx = threadIdx.x + i * FFT::stride;
        if (active && idx < NFFT) dst[idx] = thread_data[i].x * thread_data[i].x + thread_data[i].y * thread_data[i].y;
    }
}
}  // namespace f5

void solution(const double2* in, double* power, int batch, cudaStream_t stream) {
    using namespace f5;
    cudaError_t err;
    auto workspace = cufftdx::make_workspace<FFT>(err, stream);
    const unsigned blocks = (batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
    kernel<<<blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(in, power, batch, workspace);
}
