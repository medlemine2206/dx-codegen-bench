// Reference: row-major A, B, C and C = alpha*A*B + beta*C (C is read, then overwritten).
#include <cublasdx.hpp>

namespace b2 {
constexpr int M = 16, N = 16, K = 64;
using GEMM = decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() +
                      cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>() +
                      cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<BENCH_SM>());

__global__ void __launch_bounds__(GEMM::max_threads_per_block)
kernel(const float* A, const float* B, float* C, float alpha, float beta) {
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
    cublasdx::copy<GEMM, align::c>(c_g, c_s);  // beta != 0: C must be loaded
    cublasdx::copy_wait();
    GEMM().execute(alpha, a_s, b_s, beta, c_s);
    __syncthreads();
    cublasdx::copy<GEMM, align::c>(c_s, c_g);
}
}  // namespace b2

void solution(const float* A, const float* B, float* C, float alpha, float beta, int batch,
              cudaStream_t stream) {
    b2::kernel<<<batch, b2::GEMM::block_dim, cublasdx::get_shared_storage_size<b2::GEMM>(), stream>>>(
        A, B, C, alpha, beta);
}
