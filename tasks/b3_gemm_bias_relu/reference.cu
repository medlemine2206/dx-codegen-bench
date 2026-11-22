// Reference: GEMM in shared memory, then a fused epilogue (bias + ReLU) before the store.
#include <cublasdx.hpp>

namespace b3 {
constexpr int M = 64, N = 64, K = 64;
using GEMM = decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() +
                      cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
                      cublasdx::Block() + cublasdx::BlockDim<256>() + cublasdx::SM<BENCH_SM>());

__global__ void __launch_bounds__(GEMM::max_threads_per_block)
kernel(const float* A, const float* B, const float* bias, float* C) {
    extern __shared__ __align__(16) char smem[];
    const size_t id = blockIdx.x;
    auto a_g = cublasdx::make_tensor(A + id * M * K, GEMM::get_layout_gmem_a());
    auto b_g = cublasdx::make_tensor(B + id * K * N, GEMM::get_layout_gmem_b());
    auto c_g = cublasdx::make_tensor(C + id * M * N, GEMM::get_layout_gmem_c());
    auto [sa, sb, sc] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a_s = cublasdx::make_tensor(sa, GEMM::get_layout_smem_a());
    auto b_s = cublasdx::make_tensor(sb, GEMM::get_layout_smem_b());
    auto c_s = cublasdx::make_tensor(sc, GEMM::get_layout_smem_c());
    using align = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, align::a>(a_g, a_s);
    cublasdx::copy<GEMM, align::b>(b_g, b_s);
    cublasdx::copy_wait();
    GEMM().execute(1.f, a_s, b_s, 0.f, c_s);
    __syncthreads();
    for (int e = threadIdx.x; e < M * N; e += blockDim.x) {  // epilogue on the shared tile
        const int i = e % M, j = e / M;
        c_s(i, j) = fmaxf(c_s(i, j) + bias[j], 0.f);
    }
    __syncthreads();
    cublasdx::copy<GEMM, align::c>(c_s, c_g);
}
}  // namespace b3

void solution(const float* A, const float* B, const float* bias, float* C, int batch,
              cudaStream_t stream) {
    b3::kernel<<<batch, b3::GEMM::block_dim, cublasdx::get_shared_storage_size<b3::GEMM>(), stream>>>(
        A, B, bias, C);
}
