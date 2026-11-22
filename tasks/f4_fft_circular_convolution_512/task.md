# f4 — fused circular convolution via FFT

For every signal `b` in `[0, batch)`, compute the circular convolution of `x_b` with a
filter whose **spectrum** `H` is given:

    y_b = IFFT( FFT(x_b) .* H ) / N,     N = 512

where FFT uses `exp(-2*pi*i*k*n/N)`, IFFT uses `exp(+2*pi*i*k*n/N)` (unnormalised, hence
the explicit `/N`), and `.*` is the element-wise complex product.

* `x`, `y`: `batch` contiguous single-precision complex signals (`float2`), natural order.
* `H`: 512 `float2`, shared by all signals.
* `batch = 2048`.
* Forward transform, multiplication and inverse transform must happen in **one kernel**.

Implement:

```cpp
void solution(const float2* x, const float2* H, float2* y, int batch, cudaStream_t stream);
```
All pointers are device pointers. Launch your kernel(s) on `stream`.
