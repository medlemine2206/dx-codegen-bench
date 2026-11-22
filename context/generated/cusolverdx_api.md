# cusolverdx 0.2.0 — API index (generated from headers)

## Description operators (combine with `+` inside `decltype(...)`)
- `cusolverdx::Arrangement<arrangement Arr, arrangement Brr = Arr>`
- `cusolverdx::BatchesPerBlock<unsigned Value>`
- `cusolverdx::BlockDim<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>`
- `cusolverdx::Diag<diag d>`
- `cusolverdx::Block()`
- `cusolverdx::FillMode<fill_mode Value>`
- `cusolverdx::Function<function Value>`
- `cusolverdx::LeadingDimension<unsigned int LDA, unsigned int LDB = LDA>`
- `cusolverdx::PrecisionCheck<class T>`
- `cusolverdx::Precision<class PA, class PX = PA, class PB = PA>`
- `cusolverdx::Side<side s>`
- `cusolverdx::Size<unsigned int M, unsigned int N = M, unsigned int K = 1>`
- `cusolverdx::SM<unsigned int Architecture>`
- `cusolverdx::TransposeMode<transpose Value>`
- `cusolverdx::Type<type Value>`

## Enum values
- `cusolverdx::arrangement`: `col_major`, `row_major`
- `cusolverdx::diag`: `non_unit`, `unit`
- `cusolverdx::fill_mode`: `upper`, `lower`
- `cusolverdx::function`: `potrf`, `potrs`, `posv`, `getrf_no_pivot`, `getrs_no_pivot`, `gesv_no_pivot`, `getrf_partial_pivot`, `getrs_partial_pivot`, `gesv_partial_pivot`, `geqrf`, `unmqr`, `gelqf`, `unmlq`, `gels`, `trsm`
- `cusolverdx::operator_type`: `size`, `function`, `precision`, `type`, `fill_mode`, `leading_dimension`, `arrangement`, `transpose`, `diag`, `side`, `sm`, `thread`, `block`, `block_dim`, `batches_per_block`
- `cusolverdx::side`: `left`, `right`
- `cusolverdx::transpose`: `non_transposed`, `transposed`, `conj_transposed`

## Short aliases
`cusolverdx::col_major` = `arrangement::col_major`, `cusolverdx::conj_trans` = `transpose::conj_transposed`, `cusolverdx::gelqf` = `function::gelqf`, `cusolverdx::gels` = `function::gels`, `cusolverdx::geqrf` = `function::geqrf`, `cusolverdx::gesv_no_pivot` = `function::gesv_no_pivot`, `cusolverdx::gesv_partial_pivot` = `function::gesv_partial_pivot`, `cusolverdx::getrf_no_pivot` = `function::getrf_no_pivot`, `cusolverdx::getrf_partial_pivot` = `function::getrf_partial_pivot`, `cusolverdx::getrs_no_pivot` = `function::getrs_no_pivot`, `cusolverdx::getrs_partial_pivot` = `function::getrs_partial_pivot`, `cusolverdx::left` = `side::left`, `cusolverdx::left_layout` = `arrangement::col_major`, `cusolverdx::lower` = `fill_mode::lower`, `cusolverdx::non_trans` = `transpose::non_transposed`, `cusolverdx::non_unit` = `diag::non_unit`, `cusolverdx::posv` = `function::posv`, `cusolverdx::potrf` = `function::potrf`, `cusolverdx::potrs` = `function::potrs`, `cusolverdx::right` = `side::right`, `cusolverdx::right_major` = `arrangement::row_major`, `cusolverdx::row_major` = `arrangement::row_major`, `cusolverdx::trans` = `transpose::transposed`, `cusolverdx::trsm` = `function::trsm`, `cusolverdx::unit` = `diag::unit`, `cusolverdx::unmlq` = `function::unmlq`, `cusolverdx::unmqr` = `function::unmqr`, `cusolverdx::upper` = `fill_mode::upper`

## Traits available on a complete description type (e.g. `X::name`)
`a_arrangement`, `a_size`, `b_arrangement`, `b_size`, `batches_per_block`, `block_dim`, `can_be_impacted_by_nvbug_5288270`, `diag`, `fill_mode`, `is_function_cholesky`, `is_function_lu`, `is_function_lu_no_pivot`, `is_function_lu_partial_pivot`, `is_function_qr`, `is_function_solver`, `is_function_unmq`, `k_size`, `lda`, `ldb`, `m_size`, `max_threads_per_block`, `min_blocks_per_multiprocessor`, `n_size`, `shared_memory_size`, `side`, `sm`, `suggested_batches_per_block`, `suggested_block_dim`, `transpose`, `type`

## Member types
`a_cuda_data_type`, `a_data_type`, `b_cuda_data_type`, `b_data_type`, `base_type`, `status_type`, `x_cuda_data_type`, `x_data_type`

## Member functions
`execute(...)`, `get_block_dim(...)`, `get_suggested_batches_per_block(...)`, `get_suggested_block_dim(...)`, `get_thread_id(...)`

## Free functions / trait templates
`cusolverdx::arrangement_of<Description>`, `cusolverdx::block_dim_of<Description>`, `cusolverdx::function_of<Description>`, `cusolverdx::leading_dimension_of<Description>`, `cusolverdx::precision_of<Description>`, `cusolverdx::size_of<Description>`, `cusolverdx::sm_of<Description>`, `cusolverdx::type_of<Description>`
