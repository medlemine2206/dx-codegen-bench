// Reference: complex<float> GEMM. float2 is layout-compatible with cublasdx::complex<float>.
#include <cublasdx.hpp>

namespace b4 {
constexpr int M = 16, N = 16, K = 16;
using GEMM = decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::complex>() +
                      cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
                      cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<BENCH_SM>());
using value_type = typename GEMM::c_value_type;  // cublasdx::complex<float>

__global__ void __launch_bounds__(GEMM::max_threads_per_block)
kernel(const value_type* A, const value_type* B, value_type* C) {
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
    GEMM().execute(value_type{1.f, 0.f}, a_s, b_s, value_type{0.f, 0.f}, c_s);
    __syncthreads();
    cublasdx::copy<GEMM, align::c>(c_s, c_g);
}
}  // namespace b4

void solution(const float2* A, const float2* B, float2* C, int batch, cudaStream_t stream) {
    using namespace b4;
    kernel<<<batch, GEMM::block_dim, cublasdx::get_shared_storage_size<GEMM>(), stream>>>(
        reinterpret_cast<const value_type*>(A), reinterpret_cast<const value_type*>(B),
        reinterpret_cast<value_type*>(C));
}
