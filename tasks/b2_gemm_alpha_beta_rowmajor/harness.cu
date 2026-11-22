#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "b2_gemm_alpha_beta_rowmajor";
    const int M = 16, N = 16, K = 64, batch = 8192;
    const float alpha = 1.5f, beta = -0.5f;
    auto A = bench::random_floats((size_t)batch * M * K, 3);
    auto B = bench::random_floats((size_t)batch * K * N, 4);
    auto C0 = bench::random_floats((size_t)batch * M * N, 5);
    float *dA = bench::to_device(A), *dB = bench::to_device(B), *dC = bench::to_device(C0);

    solution(dA, dB, dC, alpha, beta, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dC, C0.size());

    std::vector<double> ref, out;
    for (int b = 0; b < batch; b += 11) {
        const float *a = &A[(size_t)b * M * K], *bb = &B[(size_t)b * K * N], *c0 = &C0[(size_t)b * M * N];
        std::vector<double> c(M * N);
        bench::gemm_ref(M, N, K, alpha, beta, [&](int i, int p) { return (double)a[i * K + p]; },
                        [&](int p, int j) { return (double)bb[p * N + j]; },
                        [&](int i, int j) { return (double)c0[i * N + j]; },
                        [&](int i, int j, double v) { c[i * N + j] = v; });
        for (int e = 0; e < M * N; ++e) { ref.push_back(c[e]); out.push_back(got[(size_t)b * M * N + e]); }
    }
    const double err = bench::rel_l2(ref.data(), out.data(), ref.size());
    // timing re-runs the in-place update; only speed matters there
    const double ms = bench::time_ms([&] { solution(dA, dB, dC, alpha, beta, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-5, err, ms, 2.0 * M * N * K * batch);
    return 0;
}
