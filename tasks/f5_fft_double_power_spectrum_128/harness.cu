#include SOLUTION_FILE
#include "common.cuh"

int main() {
    const char* T = "f5_fft_double_power_spectrum_128";
    const int NF = 128, batch = 8192;
    auto raw = bench::random_floats((size_t)batch * NF * 2, 31);
    std::vector<double2> in((size_t)batch * NF);
    for (size_t i = 0; i < in.size(); ++i) in[i] = make_double2(raw[2 * i], raw[2 * i + 1]);
    std::vector<double> pw(in.size(), 0.0);
    double2* din = bench::to_device(in);
    double* dp = bench::to_device(pw);

    solution(din, dp, batch, 0);
    BENCH_CHECK(T, cudaGetLastError());
    BENCH_CHECK(T, cudaDeviceSynchronize());
    auto got = bench::to_host(dp, pw.size());

    std::vector<double> ref, res;
    for (int b = 0; b < batch; b += 7) {
        std::vector<bench::cd> v(NF);
        for (int i = 0; i < NF; ++i) v[i] = {in[(size_t)b * NF + i].x, in[(size_t)b * NF + i].y};
        bench::fft_ref(v, -1);
        for (int i = 0; i < NF; ++i) { ref.push_back(std::norm(v[i])); res.push_back(got[(size_t)b * NF + i]); }
    }
    const double err = bench::rel_l2(ref.data(), res.data(), ref.size());
    const double ms = bench::time_ms([&] { solution(din, dp, batch, 0); });
    BENCH_CHECK(T, cudaGetLastError());
    bench::report(T, "ok", err < 1e-12, err, ms, 5.0 * NF * std::log2((double)NF) * batch);
    return 0;
}
