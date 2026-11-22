# cublasdx 0.4.0 — API index (generated from headers)

## Description operators (combine with `+` inside `decltype(...)`)
- `cublasdx::Alignment<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>`
- `cublasdx::Arrangement<arrangement AOrder, arrangement BOrder = arrangement::col_major, arrangement COrder = arrangement::col_major>`
- `cublasdx::FillMode<fill_mode Value>`
- `cublasdx::Function<function Value>`
- `cublasdx::LeadingDimension<unsigned int LDA, unsigned int LDB, unsigned int LDC>`
- `cublasdx::Precision<class PA, class PB = PA, class PC = PA>`
- `cublasdx::Size<unsigned int M, unsigned int N, unsigned int K = 1>`
- `cublasdx::Diagonal<diagonal Value>`
- `cublasdx::Side<side Value>`
- `cublasdx::Type<type Value>`
- `cublasdx::BlockDim<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>`
- `cublasdx::SM<unsigned int Architecture>`
- `cublasdx::Block()`

## Enum values
- `cublasdx::arrangement`: `col_major`, `row_major`
- `cublasdx::fill_mode`: `upper`, `lower`
- `cublasdx::function`: `MM`, `TRSM`
- `cublasdx::operator_type`: `size`, `precision`, `type`, `function`, `fill_mode`, `transpose_mode`, `diagonal`, `side`, `sm`, `alignment`, `arrangement`, `ld`, `block`, `thread`, `block_dim`, `experimental_tile`, `experimental_static_block_dim`
- `cublasdx::transpose_mode`: `non_transposed`, `transposed`, `conj_transposed`
- `cublasdx::diagonal`: `non_unit`, `unit`
- `cublasdx::side`: `left`, `right`

## Short aliases
`cublasdx::C` = `transpose_mode::conj_transposed`, `cublasdx::N` = `transpose_mode::non_transposed`, `cublasdx::T` = `transpose_mode::transposed`, `cublasdx::col_major` = `arrangement::col_major`, `cublasdx::left_layout` = `arrangement::col_major`, `cublasdx::right_major` = `arrangement::row_major`, `cublasdx::row_major` = `arrangement::row_major`

## Traits available on a complete description type (e.g. `X::name`)
`a_alignment`, `a_dim`, `a_size`, `b_alignment`, `b_dim`, `b_size`, `block_dim`, `c_alignment`, `c_dim`, `c_size`, `can_be_impacted_by_nvbug_5218000`, `condition`, `lda`, `ldb`, `ldc`, `max_threads_per_block`, `min_blocks_per_multiprocessor`, `suggested_block_dim`

## Member types
`a_type`, `a_value_type`, `b_type`, `b_value_type`, `base_type`, `c_type`, `c_value_type`, `db_a_type`, `db_b_type`, `db_c_type`, `description_type`, `execution_type`, `operator_wrapper_type`

## Member functions
`execute(...)`, `get_block_dim(...)`, `get_cute_layout(...)`, `get_layout_gmem_a(...)`, `get_layout_gmem_b(...)`, `get_layout_gmem_c(...)`, `get_layout_smem_a(...)`, `get_layout_smem_b(...)`, `get_layout_smem_c(...)`, `get_partitioner(...)`, `get_suggested_block_dim(...)`, `suggest_cute_layout_smem(...)`, `suggest_layout_smem_a(...)`, `suggest_layout_smem_b(...)`, `suggest_layout_smem_c(...)`, `suggest_partitioner(...)`

## Free functions / trait templates
`cublasdx::alignment_of<Description>`, `cublasdx::copy`, `cublasdx::copy_wait`, `cublasdx::leading_dimension_of<Description>`, `cublasdx::slice_shared_memory`, `cublasdx::slice_shared_memory_ab`
