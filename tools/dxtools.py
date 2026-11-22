"""Shared helpers: locate the toolchain, compile/run a task, compile-check a snippet.

Used by the evaluation runner (eval/run_eval.py), the MCP server (tools/mcp_server.py)
and the documentation test (python tools/dxtools.py check-docs).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASKS_DIR = os.path.join(ROOT, "tasks")
CONTEXT_DIR = os.path.join(ROOT, "context")
HARNESS_DIR = os.path.join(ROOT, "harness")


def find_mathdx():
    env = os.environ.get("MATHDX_ROOT")
    if env and os.path.isfile(os.path.join(env, "include", "cublasdx.hpp")):
        return env
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations:
        for base in spec.submodule_search_locations:
            root = os.path.join(base, "mathdx")
            if os.path.isfile(os.path.join(root, "include", "cublasdx.hpp")):
                return root
    raise RuntimeError("MathDx not found: pip install nvidia-mathdx==25.6.0 or set MATHDX_ROOT")


def find_nvcc():
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc and os.path.exists("/usr/local/cuda/bin/nvcc"):
        nvcc = "/usr/local/cuda/bin/nvcc"
    if not nvcc:
        raise RuntimeError("nvcc not found (set NVCC=/path/to/nvcc)")
    return nvcc


def detect_arch(default="75"):
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                      text=True, timeout=20)
        return out.split()[0].replace(".", "")
    except Exception:
        return os.environ.get("BENCH_ARCH", default)


def nvcc_flags(arch):
    mdx = find_mathdx()
    return ["-std=c++17", "-O3", "--expt-relaxed-constexpr", f"-arch=sm_{arch}", f"-DBENCH_SM={arch}0",
            "-I" + os.path.join(mdx, "include"), "-I" + os.path.join(mdx, "external", "cutlass", "include"),
            "-I" + HARNESS_DIR, "-w"]


def list_tasks():
    return sorted(d for d in os.listdir(TASKS_DIR) if os.path.isdir(os.path.join(TASKS_DIR, d)))


def task_spec(task):
    with open(os.path.join(TASKS_DIR, task, "task.md")) as f:
        return f.read()


def _trim(log, limit=6000):
    """Keep compiler output readable: drop noise, keep the first errors."""
    lines = [l for l in log.splitlines() if "deprecated" not in l]
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def compile_task(task, solution_path, arch, out_bin, timeout=600):
    """Compile tasks/<task>/harness.cu against a solution file. Returns (ok, log)."""
    cmd = [find_nvcc()] + nvcc_flags(arch) + [f'-DSOLUTION_FILE="{os.path.abspath(solution_path)}"',
                                              os.path.join(TASKS_DIR, task, "harness.cu"), "-o", out_bin]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "compilation timed out"
    return p.returncode == 0, _trim(p.stdout + p.stderr)


def run_binary(binary, timeout=120):
    """Run a compiled harness and parse its RESULT line."""
    try:
        p = subprocess.run([binary], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "correct": False, "message": f"no result after {timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"status": "crash", "correct": False,
            "message": _trim((p.stdout + p.stderr)[-2000:] or f"exit code {p.returncode}", 2000)}


def compile_check(code, arch=None, timeout=300):
    """Compile a CUDA snippet to an object file (no linking, no GPU needed). Returns (ok, log)."""
    arch = arch or detect_arch()
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "snippet.cu")
        with open(src, "w") as f:
            f.write(code)
        cmd = [find_nvcc()] + nvcc_flags(arch) + ["-c", src, "-o", os.path.join(d, "snippet.o")]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "compilation timed out"
        log = _trim(p.stdout + p.stderr).replace(src, "solution.cu")
        return p.returncode == 0, log


def extract_code(text):
    """Pull the C++/CUDA code out of an LLM answer (largest fenced block, else raw text)."""
    blocks = re.findall(r"```(?:[a-zA-Z+]*)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def check_docs(arch="75"):
    """Compile every ```cuda compile-check block of the context pack."""
    failures = 0
    for fn in sorted(os.listdir(CONTEXT_DIR)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(CONTEXT_DIR, fn)) as f:
            text = f.read()
        for i, block in enumerate(re.findall(r"```cuda compile-check\n(.*?)```", text, flags=re.S)):
            ok, log = compile_check(block, arch)
            print(f"{'OK  ' if ok else 'FAIL'} {fn} block {i}")
            if not ok:
                failures += 1
                print(log[:3000])
    return failures


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check-docs":
        sys.exit(1 if check_docs(sys.argv[2] if len(sys.argv) > 2 else detect_arch()) else 0)
    print("usage: python tools/dxtools.py check-docs [arch]")
