# b1 — batched small GEMM

For every batch index `b` in `[0, batch)`: `C_b = A_b * B_b`.

* `A_b`, `B_b`, `C_b`: 32 x 32 `float`, **column-major**, stored contiguously one after
  the other (`A_b` starts at `A + b*32*32`, same for `B` and `C`).
* `batch = 4096`.

Implement:

```cpp
void solution(const float* A, const float* B, float* C, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
