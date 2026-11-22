#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "f2_fft_inverse_normalized_1024";
    const int NF = 1024, batch = 1024;
    auto raw = bench::random_floats((size_t)batch * NF * 2, 12);
    std::vector<float2> in((size_t)batch * NF), out(in.size());
    for (size_t i = 0; i < in.size(); ++i) in[i] = make_float2(raw[2 * i], raw[2 * i + 1]);
    float2 *din = bench::to_device(in), *dout = bench::to_device(out);

    solution(din, dout, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dout, out.size());

    std::vector<double> ref, res;
    const int step = batch > 4096 ? 61 : 5;
    for (int b = 0; b < batch; b += step) {
        std::vector<bench::cd> x(NF);
        for (int i = 0; i < NF; ++i) x[i] = {in[(size_t)b * NF + i].x, in[(size_t)b * NF + i].y};
        bench::fft_ref(x, 1);
        for (int i = 0; i < NF; ++i) {
            const bench::cd v = x[i] / (double)NF;
            ref.push_back(v.real()); ref.push_back(v.imag());
            res.push_back(got[(size_t)b * NF + i].x); res.push_back(got[(size_t)b * NF + i].y);
        }
    }
    const double err = bench::rel_l2(ref.data(), res.data(), ref.size());
    const double ms = bench::time_ms([&] { solution(din, dout, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-5, err, ms, 5.0 * NF * std::log2((double)NF) * batch);
    return 0;
}
