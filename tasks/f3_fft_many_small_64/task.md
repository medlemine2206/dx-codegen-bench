# f3 — many small forward FFTs

For every signal `b` in `[0, batch)`:

    out_b[k] = sum_n in_b[n] * exp(-2*pi*i*k*n/N),   N = 64

* Single-precision complex (`float2`), contiguous signals (`in + b*64`), natural order.
* `batch = 65536` — throughput matters: the score includes the speed relative to a
  reference implementation.

Implement:

```cpp
void solution(const float2* in, float2* out, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
