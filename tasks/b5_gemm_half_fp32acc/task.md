# b5 — batched mixed-precision GEMM (half in, float out)

For every batch index `b` in `[0, batch)`: `C_b = A_b * B_b`.

* `A_b`, `B_b`: 64 x 64 `__half`, `C_b`: 64 x 64 `float`, all **column-major**,
  stored contiguously.
* Products must be **accumulated in float** (result accuracy is checked against a
  float/double accumulation of the half inputs; tolerance 1e-4 relative L2).
* `batch = 1024`.

Implement:

```cpp
void solution(const __half* A, const __half* B, float* C, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
