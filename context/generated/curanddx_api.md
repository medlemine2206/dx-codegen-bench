# curanddx 0.2.0 — API index (generated from headers)

## Description operators (combine with `+` inside `decltype(...)`)
- `curanddx::Generator<generator Value>`
- `curanddx::Ordering<order Value>`
- `curanddx::PhiloxRounds<unsigned int Value>`
- `curanddx::BlockDim<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>`
- `curanddx::GridDim<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>`
- `curanddx::SM<unsigned int Architecture>`
- `curanddx::Thread()`
- `curanddx::Block()`
- `curanddx::Device()`

## Enum values
- `curanddx::generator`: `xorwow`, `mrg32k3a`, `philox4_32`, `mtgp32`, `pcg`, `sobol32`, `scrambled_sobol32`, `sobol64`, `scrambled_sobol64`
- `curanddx::operator_type`: `generator`, `philox_rounds`, `sm`, `thread`, `block`, `device`, `block_dim`, `grid_dim`, `ordering`
- `curanddx::order`: `strict`, `curand_legacy`, `subsequence`

## Short aliases
`curanddx::curand_legacy` = `order::curand_legacy`, `curanddx::mrg32k3a` = `generator::mrg32k3a`, `curanddx::mtgp32` = `generator::mtgp32`, `curanddx::pcg` = `generator::pcg`, `curanddx::philox4_32` = `generator::philox4_32`, `curanddx::scrambled_sobol32` = `generator::scrambled_sobol32`, `curanddx::scrambled_sobol64` = `generator::scrambled_sobol64`, `curanddx::sobol32` = `generator::sobol32`, `curanddx::sobol64` = `generator::sobol64`, `curanddx::strict` = `order::strict`, `curanddx::subsequence` = `order::subsequence`, `curanddx::xorwow` = `generator::xorwow`
