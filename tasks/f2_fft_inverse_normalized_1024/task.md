# f2 — batched inverse FFT, normalised

For every signal `b` in `[0, batch)`:

    out_b[n] = (1/N) * sum_k in_b[k] * exp(+2*pi*i*k*n/N),   N = 1024

* Signals are single-precision complex (`float2`, `.x` real, `.y` imaginary), stored
  contiguously: signal `b` starts at `in + b*1024`.
* Output in natural order, same layout.
* `batch = 1024`.

Implement:

```cpp
void solution(const float2* in, float2* out, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
