# cufftdx 1.5.0 — API index (generated from headers)

## Description operators (combine with `+` inside `decltype(...)`)
- `cufftdx::FFTsPerBlock<unsigned int Value>`
- `cufftdx::ElementsPerThread<unsigned int Value>`
- `cufftdx::Direction<fft_direction Value>`
- `cufftdx::Precision<class T>`
- `cufftdx::RealFFTOptions<complex_layout RealLayout, real_mode RealMode = real_mode::normal>`
- `cufftdx::Size<unsigned int Value>`
- `cufftdx::Type<fft_type Value>`
- `cufftdx::BlockDim<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>`
- `cufftdx::SM<unsigned int Architecture>`
- `cufftdx::Thread()`
- `cufftdx::Block()`

## Enum values
- `cufftdx::fft_direction`: `forward`, `inverse`
- `cufftdx::fft_operator`: `direction`, `precision`, `size`, `sm`, `type`, `real_fft_options`, `thread`, `block`, `elements_per_thread`, `ffts_per_block`, `block_dim`
- `cufftdx::complex_layout`: `natural`, `packed`, `full`
- `cufftdx::real_mode`: `normal`, `folded`, `batch_optimized`
- `cufftdx::fft_type`: `c2c`, `c2r`, `r2c`

## Traits available on a complete description type (e.g. `X::name`)
`block_dim`, `condition`, `effective_elements_per_thread`, `elements_per_thread`, `fft_blue_size`, `fft_execution_size`, `fft_memory_size`, `ffts_per_block`, `function_id`, `implicit_type_batching`, `input_ept`, `input_length`, `is_block_execution`, `is_folded`, `is_redistributable`, `is_this_fft_max_block_size`, `is_thread_execution`, `max_block_size`, `max_meaningful_ept`, `max_threads_per_block`, `output_ept`, `output_length`, `requires_redistribution`, `requires_workspace`, `shared_memory_size`, `storage_size`, `store_ept`, `stride`, `suggested_elements_per_thread`, `suggested_ffts_per_block`, `use_bluestein`, `workspace_size`

## Member types
`base_type`, `description_type`, `execution_type`, `fft_block_execution_type`, `fft_thread_execution_type`, `host_workspace_type`, `input_type`, `ldg_type`, `operator_wrapper_type`, `output_type`, `scalar_type`, `value_type`, `workspace_type`

## Member functions
`execute(...)`, `get_input_ept(...)`, `get_input_length(...)`, `get_output_ept(...)`, `get_output_length(...)`

## Free functions / trait templates
`cufftdx::direction_of<Description>`, `cufftdx::make_workspace`, `cufftdx::precision_of<Description>`, `cufftdx::size_of<Description>`, `cufftdx::sm_of<Description>`, `cufftdx::type_of<Description>`
