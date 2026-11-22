#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "b5_gemm_half_fp32acc";
    const int M = 64, N = 64, K = 64, batch = 1024;
    auto fa = bench::random_floats((size_t)batch * M * K, 12);
    auto fb = bench::random_floats((size_t)batch * K * N, 13);
    std::vector<__half> A(fa.size()), B(fb.size());
    std::vector<float> Ar(fa.size()), Br(fb.size());  // inputs as actually seen by the GPU
    for (size_t i = 0; i < fa.size(); ++i) { A[i] = __float2half(fa[i]); Ar[i] = __half2float(A[i]); }
    for (size_t i = 0; i < fb.size(); ++i) { B[i] = __float2half(fb[i]); Br[i] = __half2float(B[i]); }
    std::vector<float> C((size_t)batch * M * N, 0.f);
    __half *dA = bench::to_device(A), *dB = bench::to_device(B);
    float* dC = bench::to_device(C);

    solution(dA, dB, dC, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dC, C.size());

    std::vector<double> ref, out;
    for (int b = 0; b < batch; b += 13) {
        const float *a = &Ar[(size_t)b * M * K], *bb = &Br[(size_t)b * K * N];
        std::vector<double> c(M * N);
        bench::gemm_ref(M, N, K, 1.0, 0.0, [&](int i, int p) { return (double)a[p * M + i]; },
                        [&](int p, int j) { return (double)bb[j * K + p]; }, [&](int, int) { return 0.0; },
                        [&](int i, int j, double v) { c[j * M + i] = v; });
        for (int e = 0; e < M * N; ++e) { ref.push_back(c[e]); out.push_back(got[(size_t)b * M * N + e]); }
    }
    const double err = bench::rel_l2(ref.data(), out.data(), ref.size());
    const double ms = bench::time_ms([&] { solution(dA, dB, dC, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-4, err, ms, 2.0 * M * N * K * batch);
    return 0;
}
