# b4 — batched complex GEMM

For every batch index `b` in `[0, batch)`: `C_b = A_b * B_b` with complex numbers.

* `A_b`, `B_b`, `C_b`: 16 x 16 single-precision **complex** matrices, **column-major**,
  stored contiguously. Each complex number is a `float2` (`.x` = real, `.y` = imaginary).
* `batch = 8192`.

Implement:

```cpp
void solution(const float2* A, const float2* B, float2* C, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
