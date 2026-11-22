#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "f1_fft_c2c_256";
    const int NF = 256, batch = 4096;
    auto raw = bench::random_floats((size_t)batch * NF * 2, 11);
    std::vector<float2> in((size_t)batch * NF), out(in.size());
    for (size_t i = 0; i < in.size(); ++i) in[i] = make_float2(raw[2 * i], raw[2 * i + 1]);
    float2 *din = bench::to_device(in), *dout = bench::to_device(out);

    solution(din, dout, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dout, out.size());

    std::vector<double> ref, res;
    for (int b = 0; b < batch; b += 5) {
        std::vector<bench::cd> x(NF);
        for (int i = 0; i < NF; ++i) x[i] = {in[(size_t)b * NF + i].x, in[(size_t)b * NF + i].y};
        bench::fft_ref(x, -1);
        for (int i = 0; i < NF; ++i) {
            ref.push_back(x[i].real()); ref.push_back(x[i].imag());
            res.push_back(got[(size_t)b * NF + i].x); res.push_back(got[(size_t)b * NF + i].y);
        }
    }
    const double err = bench::rel_l2(ref.data(), res.data(), ref.size());
    const double ms = bench::time_ms([&] { solution(din, dout, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-5, err, ms, 5.0 * NF * std::log2((double)NF) * batch);
    return 0;
}
