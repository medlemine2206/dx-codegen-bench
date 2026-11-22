// Reference solution: one thread block per matrix, the canonical cuBLASDx flow
// global -> shared (cublasdx::copy), BLAS::execute, shared -> global.
#include <cublasdx.hpp>

namespace b1 {
constexpr int M = 32, N = 32, K = 32;

using GEMM = decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() +
                      cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
                      cublasdx::Block() + cublasdx::BlockDim<256>() + cublasdx::SM<BENCH_SM>());

__global__ void __launch_bounds__(GEMM::max_threads_per_block)
kernel(const float* A, const float* B, float* C) {
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
    cublasdx::copy<GEMM, align::c>(c_s, c_g);
}
}  // namespace b1

void solution(const float* A, const float* B, float* C, int batch, cudaStream_t stream) {
    const auto smem = cublasdx::get_shared_storage_size<b1::GEMM>();
    b1::kernel<<<batch, b1::GEMM::block_dim, smem, stream>>>(A, B, C);
}
