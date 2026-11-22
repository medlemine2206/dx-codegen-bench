// common.cuh — shared test harness: data generation, CPU references, checks, timing.
//
// Every task harness does:
//     #include SOLUTION_FILE          // the code under test (defines `solution(...)`)
//     #include "common.cuh"
// and prints exactly one line starting with "RESULT " followed by JSON, which the
// evaluation runner parses.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace bench {

inline void report(const char* task, const char* status, bool correct, double rel_err, double ms,
                   double flops, const std::string& msg = "") {
    std::printf("RESULT {\"task\": \"%s\", \"status\": \"%s\", \"correct\": %s, \"rel_err\": %.3e, "
                "\"time_ms\": %.6f, \"gflops\": %.3f, \"message\": \"%s\"}\n",
                task, status, correct ? "true" : "false", rel_err, ms,
                (ms > 0 ? flops / (ms * 1e6) : 0.0), msg.c_str());
    std::fflush(stdout);
}

// Abort with a RESULT line on any CUDA error (illegal address, bad launch config, ...).
#define BENCH_CHECK(task, call)                                                              \
    do {                                                                                     \
        cudaError_t e__ = (call);                                                            \
        if (e__ != cudaSuccess) {                                                            \
            bench::report(task, "runtime_error", false, 0, 0, 0, cudaGetErrorString(e__));   \
            std::exit(2);                                                                    \
        }                                                                                    \
    } while (0)

inline std::vector<float> random_floats(size_t n, uint64_t seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = u(rng);
    return v;
}

inline double rel_l2(const double* ref, const double* got, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = got[i] - ref[i];
        if (!std::isfinite(got[i])) return INFINITY;
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1.0));
}

// ---- CPU references (double precision) -------------------------------------------
// C = alpha * A * B + beta * C, element accessors let each task pick its layout.
template <class GetA, class GetB, class GetC, class SetC>
void gemm_ref(int m, int n, int k, double alpha, double beta, GetA a, GetB b, GetC c, SetC set) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0;
            for (int p = 0; p < k; ++p) acc += a(i, p) * b(p, j);
            set(i, j, alpha * acc + beta * c(i, j));
        }
}

using cd = std::complex<double>;

// In-place radix-2 FFT, sign = -1 forward, +1 inverse (unnormalised), n power of two.
inline void fft_ref(std::vector<cd>& a, int sign) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2 * M_PI / (double)len;
        const cd wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cd w(1);
            for (size_t j = 0; j < len / 2; ++j) {
                const cd u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

// ---- timing: median of `reps` launches -------------------------------------------
template <class F>
double time_ms(F&& f, int reps = 20) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<float> t;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

template <class T>
T* to_device(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <class T>
std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

}  // namespace bench
