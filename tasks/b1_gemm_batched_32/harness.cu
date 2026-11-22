#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "b1_gemm_batched_32";
    const int M = 32, N = 32, K = 32, batch = 4096;
    auto A = bench::random_floats((size_t)batch * M * K, 1);
    auto B = bench::random_floats((size_t)batch * K * N, 2);
    std::vector<float> C((size_t)batch * M * N, 0.f);
    float *dA = bench::to_device(A), *dB = bench::to_device(B), *dC = bench::to_device(C);

    solution(dA, dB, dC, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dC, C.size());

    std::vector<double> ref, out;
    for (int b = 0; b < batch; b += 7) {  // check a spread-out subset of the batch
        const float *a = &A[(size_t)b * M * K], *bb = &B[(size_t)b * K * N];
        std::vector<double> c(M * N, 0.0);
        bench::gemm_ref(M, N, K, 1.0, 0.0, [&](int i, int p) { return (double)a[p * M + i]; },
                        [&](int p, int j) { return (double)bb[j * K + p]; }, [&](int, int) { return 0.0; },
                        [&](int i, int j, double v) { c[j * M + i] = v; });
        for (int e = 0; e < M * N; ++e) { ref.push_back(c[e]); out.push_back(got[(size_t)b * M * N + e]); }
    }
    const double err = bench::rel_l2(ref.data(), out.data(), ref.size());
    const double ms = bench::time_ms([&] { solution(dA, dB, dC, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-5, err, ms, 2.0 * M * N * K * batch);
    return 0;
}
