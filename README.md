# Can an LLM write cuBLASDx / cuFFTDx code? A benchmark and an LLM-friendly context pack

NVIDIA's device-side math libraries (**cuBLASDx, cuFFTDx**, cuSolverDx, ...) are recent,
header-only, template-heavy C++ APIs. Code assistants have seen little code using them, so
they tend to invent operators, launch kernels with the wrong block shape or return early
before a block-wide `execute()`.

This repository measures that problem and tests one way to fix it:

1. **A benchmark**: 10 tasks (5 cuBLASDx, 5 cuFFTDx). Each generated `.cu` file is
   compiled, run on the GPU and checked against a double-precision CPU reference, and its
   speed is compared with a reference solution.
2. **A context pack** that turns the library API into something an LLM can use:
   a short curated guide, a pitfalls checklist and an **API index extracted automatically
   from the installed headers**. All code examples are compile-tested.
3. **An MCP server** that gives any coding assistant (Claude Code, Claude Desktop, ...)
   the docs, a search tool and a `compile_check` tool backed by `nvcc`.
4. **An evaluation runner** that compares three conditions and reports pass rates with
   confidence intervals, plus a breakdown of what fails.

| Condition | What the model gets | Prompt size |
|---|---|---|
| `baseline` | task description only | ≈ 0.3 k tokens |
| `context` | + context pack (guide, pitfalls, generated API index) | ≈ 3.7 k tokens |
| `context_feedback` | + up to 3 repair rounds with compiler errors / test output | 3.7 k tokens + feedback |

---

## Tasks

| id | library | what it tests |
|---|---|---|
| `b1_gemm_batched_32` | cuBLASDx | the canonical flow: description, tensors, `copy`, `execute`, launch config |
| `b2_gemm_alpha_beta_rowmajor` | cuBLASDx | `Arrangement<row_major,...>`, loading C when β ≠ 0 |
| `b3_gemm_bias_relu` | cuBLASDx | fused epilogue on the shared-memory tile |
| `b4_gemm_complex` | cuBLASDx | `Type<complex>`, `float2` ↔ `cublasdx::complex<float>` |
| `b5_gemm_half_fp32acc` | cuBLASDx | mixed precision `Precision<__half, __half, float>` |
| `f1_fft_c2c_256` | cuFFTDx | data distribution across threads, `FFTsPerBlock`, workspace |
| `f2_fft_inverse_normalized_1024` | cuFFTDx | inverse transforms are not normalised |
| `f3_fft_many_small_64` | cuFFTDx | throughput: packing many tiny FFTs per block |
| `f4_fft_circular_convolution_512` | cuFFTDx | forward FFT → pointwise product → inverse FFT in one kernel |
| `f5_fft_double_power_spectrum_128` | cuFFTDx | double precision + fused epilogue |

Each task folder contains `task.md` (the prompt), `harness.cu` (data generation, CPU
reference, check, timing) and `reference.cu` (a working solution, used to validate the
harness and as the speed baseline).

A sample **passes** if it compiles, runs without error, matches the CPU reference
(relative L2 error below 1e-5 in float, 1e-4 for half inputs, 1e-12 in double) **and**
really calls the Dx library's `execute()` inside a kernel. A hand-written FFT that happens
to be correct does not count.

## The context pack (`context/`)

* `00_overview.md` — the one pattern shared by every Dx library (describe → read the
  launch config from the description → launch → `execute()` by all threads), and the
  build line.
* `cublasdx.md`, `cufftdx.md` — task-oriented guides with a complete canonical kernel each.
  The examples are compiled by `python tools/dxtools.py check-docs`, so the docs cannot
  drift away from the real API.
* `pitfalls.md` — the mistakes that break compilation or results (early `return` before
  `execute()`, wrong block shape, unnormalised inverse FFT, ...).
* `generated/*_api.md|json` — **generated** by `tools/extract_api.py` from the headers
  actually installed: every operator with its template parameters, enum values, traits
  (`FFT::block_dim`, `BLAS::a_size`, ...) and free functions. When the library is
  updated, rerunning the script updates the index.

## MCP server

```bash
pip install -r requirements.txt
claude mcp add dx-docs -- python /abs/path/to/tools/mcp_server.py   # Claude Code
```
Tools: `list_docs`, `read_doc`, `search_docs`, `compile_check(code, arch)`. The last one
lets an assistant check its own code with `nvcc` before answering. That is the same loop
as the `context_feedback` condition, which the benchmark measures.

## Results

> Run `notebooks/run_on_colab.ipynb` (free T4 + an API key), then paste
> `results/report/summary.md` and the plots here.

| model | baseline | + context pack | + context + compiler feedback |
|---|---|---|---|
| | | | |

<!-- after the run: ![pass rates](results/report/pass_rates_claude-haiku-5-5.png) -->

## Run it

```bash
pip install -r requirements.txt                 # MathDx 25.06 headers, anthropic, mcp, matplotlib
python tools/extract_api.py                     # regenerate the API index from the headers
python tools/dxtools.py check-docs              # compile every documented example
python eval/run_eval.py --oracle                # validate the 10 harnesses, reference timings
export ANTHROPIC_API_KEY=...
python eval/run_eval.py --model claude-haiku-5-5 --samples 3
python eval/report.py results/*.jsonl --out results/report
```
Useful options: `--tasks b1,f4` (prefixes), `--conditions baseline,context`,
`--max-repairs 3`, `--provider openai --model <name>`, `--compile-only` (no GPU),
`--provider mock` (tests the whole pipeline with the reference solutions, no API key).

Every generated file is kept in `results/<run>/code/` and every attempt (category,
compiler log excerpt, error, timing) in the `.jsonl`, so each failure can be read in full.

## Code map

```
tasks/<id>/{task.md, harness.cu, reference.cu}   benchmark tasks
harness/common.cuh                               CPU references (GEMM, radix-2 FFT), checks, timing
context/                                         curated guides, pitfalls, generated API index
tools/extract_api.py                             header -> API index (markdown + JSON)
tools/dxtools.py                                 compile / run / compile_check / doc test
tools/mcp_server.py                              MCP server (docs + search + compiler)
eval/prompts.py, eval/run_eval.py                conditions, LLM calls, scoring, failure taxonomy
eval/report.py                                   pass@1 with Wilson CIs, pass@3, per-task heatmap, failures
notebooks/run_on_colab.ipynb                     end-to-end run on a free T4
```

## Limitations and next steps

* 10 tasks × a few samples gives wide confidence intervals: the report shows them.
  More tasks (cuSolverDx `posv`, cuRANDDx, GEMM + FFT fusion) are the next step.
* Ablations to run: generated API index only vs curated guide only vs pitfalls only, to
  measure which part of the context actually helps.
* Retrieval (MCP `search_docs`) vs putting the whole pack in the prompt: same pass rate
  for fewer tokens?
* The pack documents MathDx 25.06. Newer releases change parts of the API; regenerating
  the index handles that, the curated guide has to be reviewed.
