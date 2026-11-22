"""Prompt construction for the three evaluation conditions."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import dxtools  # noqa: E402

SYSTEM_BASE = """You are an expert CUDA C++ engineer.
Write ONE self-contained CUDA source file that implements the requested `solution(...)`
host function and the kernel(s) it launches.

Hard requirements:
- Use NVIDIA MathDx device-side libraries INSIDE your kernel: cuBLASDx (`#include <cublasdx.hpp>`)
  for matrix-multiplication tasks, cuFFTDx (`#include <cufftdx.hpp>`) for FFT tasks.
  Solutions that do not call the library's `execute()` in a kernel are rejected.
- Define `solution` with exactly the given signature. Do not define `main()`.
- The file is compiled as:
    nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_XY -DBENCH_SM=XY0 \\
         -I<mathdx>/include -I<mathdx>/external/cutlass/include harness.cu
  where harness.cu #includes your file. The macro BENCH_SM (e.g. 750 for sm_75) is defined.
  cuda_runtime.h and cuda_fp16.h are already included.
- Results are checked against a double-precision CPU reference; speed is measured.
- Answer with a single ```cuda code block containing the whole file, nothing else."""

CONTEXT_HEADER = """
Below is reference documentation for the MathDx libraries (written for the exact
versions installed). Read it carefully before writing code.
"""

CONDITIONS = {
    "baseline": "task description only",
    "context": "task + context pack (curated guide, pitfalls, API index generated from headers)",
    "context_feedback": "context pack + up to N repair rounds with compiler / test feedback",
}


def library_of(task):
    return "cublasdx" if task.startswith("b") else "cufftdx"


def context_pack(task):
    lib = library_of(task)
    files = ["00_overview.md", f"{lib}.md", "pitfalls.md", os.path.join("generated", f"{lib}_api.md")]
    parts = []
    for fn in files:
        path = os.path.join(dxtools.CONTEXT_DIR, fn)
        if os.path.exists(path):
            with open(path) as f:
                parts.append(f"<document name=\"{fn}\">\n{f.read()}\n</document>")
    return "\n\n".join(parts)


def system_prompt(task, condition):
    if condition == "baseline":
        return SYSTEM_BASE
    return SYSTEM_BASE + "\n" + CONTEXT_HEADER + "\n" + context_pack(task)


def task_prompt(task):
    return "Task:\n\n" + dxtools.task_spec(task)


def feedback_prompt(stage, details):
    if stage == "compile":
        head = "Your file does not compile. Compiler output:"
    elif stage == "no_library":
        head = ("Your solution does not use the required MathDx device library inside a kernel "
                "(no cuBLASDx/cuFFTDx `execute()` call found).")
    elif stage == "wrong_result":
        head = "Your file compiles and runs, but the result is wrong. Test output:"
    else:
        head = "Your file compiles but fails at run time. Test output:"
    return (f"{head}\n\n```\n{details}\n```\n\nFix the problem and answer with the complete corrected "
            "file in a single ```cuda code block.")
