# f5 — double-precision FFT + power spectrum

For every signal `b` in `[0, batch)`:

    X_b[k] = sum_n in_b[n] * exp(-2*pi*i*k*n/N),   power_b[k] = |X_b[k]|^2,   N = 128

* Input: double-precision complex (`double2`), contiguous signals (`in + b*128`).
* Output: `power`, `batch * 128` doubles, natural order.
* Must be computed in **double precision** (tolerance 1e-12 relative L2).
* `batch = 8192`.

Implement:

```cpp
void solution(const double2* in, double* power, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
