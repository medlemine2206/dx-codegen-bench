# f1 — batched forward FFT

For every signal `b` in `[0, batch)`:

    out_b[k] = sum_n in_b[n] * exp(-2*pi*i*k*n/N),   N = 256

* Single-precision complex (`float2`, `.x` real, `.y` imaginary), signals stored
  contiguously (`in + b*256`), output in natural order with the same layout.
* `batch = 4096`.

Implement:

```cpp
void solution(const float2* in, float2* out, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
