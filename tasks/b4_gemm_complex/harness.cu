#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "b4_gemm_complex";
    const int M = 16, N = 16, K = 16, batch = 8192;
    auto ra = bench::random_floats((size_t)batch * M * K * 2, 9);
    auto rb = bench::random_floats((size_t)batch * K * N * 2, 10);
    std::vector<float2> A((size_t)batch * M * K), B((size_t)batch * K * N), C((size_t)batch * M * N);
    for (size_t i = 0; i < A.size(); ++i) A[i] = make_float2(ra[2 * i], ra[2 * i + 1]);
    for (size_t i = 0; i < B.size(); ++i) B[i] = make_float2(rb[2 * i], rb[2 * i + 1]);
    float2 *dA = bench::to_device(A), *dB = bench::to_device(B), *dC = bench::to_device(C);

    solution(dA, dB, dC, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dC, C.size());

    std::vector<double> ref, out;
    for (int b = 0; b < batch; b += 17) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) {
                bench::cd acc = 0;
                for (int p = 0; p < K; ++p) {
                    const float2 a = A[(size_t)b * M * K + p * M + i], x = B[(size_t)b * K * N + j * K + p];
                    acc += bench::cd(a.x, a.y) * bench::cd(x.x, x.y);
                }
                const float2 g = got[(size_t)b * M * N + j * M + i];
                ref.push_back(acc.real()); ref.push_back(acc.imag());
                out.push_back(g.x); out.push_back(g.y);
            }
    }
    const double err = bench::rel_l2(ref.data(), out.data(), ref.size());
    const double ms = bench::time_ms([&] { solution(dA, dB, dC, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-5, err, ms, 8.0 * M * N * K * batch);
    return 0;
}
