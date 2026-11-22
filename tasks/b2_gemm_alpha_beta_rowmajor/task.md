# b2 — batched GEMM, row-major, alpha/beta update

For every batch index `b` in `[0, batch)`, compute **in place**

    C_b = alpha * A_b * B_b + beta * C_b

* `A_b`: 16 x 64, `B_b`: 64 x 16, `C_b`: 16 x 16, all `float`, **row-major**, stored
  contiguously one after the other (`A_b` starts at `A + b*16*64`, etc.).
* `C` holds meaningful input values (beta != 0).
* `batch = 8192`.

Implement:

```cpp
void solution(const float* A, const float* B, float* C, float alpha, float beta,
              int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
