# b3 — batched GEMM with fused bias + ReLU epilogue

For every batch index `b` in `[0, batch)`:

    C_b[i][j] = max( (A_b * B_b)[i][j] + bias[j], 0 )

* `A_b`, `B_b`, `C_b`: 64 x 64 `float`, **column-major**, stored contiguously
  (`A_b` starts at `A + b*64*64`).
* `bias`: 64 floats, one per **column** j, shared by all batch entries.
* `batch = 1024`.
* The bias and ReLU must be applied inside the same kernel as the matrix product
  (no separate element-wise kernel).

Implement:

```cpp
void solution(const float* A, const float* B, const float* bias, float* C,
              int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
