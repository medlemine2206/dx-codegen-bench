#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "f4_fft_circular_convolution_512";
    const int NF = 512, batch = 2048;
    auto rx = bench::random_floats((size_t)batch * NF * 2, 21);
    auto rh = bench::random_floats(NF * 2, 22);
    std::vector<float2> x((size_t)batch * NF), H(NF), y(x.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = make_float2(rx[2 * i], rx[2 * i + 1]);
    for (int i = 0; i < NF; ++i) H[i] = make_float2(rh[2 * i], rh[2 * i + 1]);
    float2 *dx = bench::to_device(x), *dH = bench::to_device(H), *dy = bench::to_device(y);

    solution(dx, dH, dy, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dy, y.size());

    std::vector<double> ref, res;
    for (int b = 0; b < batch; b += 9) {
        std::vector<bench::cd> v(NF);
        for (int i = 0; i < NF; ++i) v[i] = {x[(size_t)b * NF + i].x, x[(size_t)b * NF + i].y};
        bench::fft_ref(v, -1);
        for (int i = 0; i < NF; ++i) v[i] *= bench::cd(H[i].x, H[i].y);
        bench::fft_ref(v, +1);
        for (int i = 0; i < NF; ++i) {
            v[i] /= (double)NF;
            ref.push_back(v[i].real()); ref.push_back(v[i].imag());
            res.push_back(got[(size_t)b * NF + i].x); res.push_back(got[(size_t)b * NF + i].y);
        }
    }
    const double err = bench::rel_l2(ref.data(), res.data(), ref.size());
    const double ms = bench::time_ms([&] { solution(dx, dH, dy, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-5, err, ms, 2 * 5.0 * NF * std::log2((double)NF) * batch);
    return 0;
}
