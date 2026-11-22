"""Evaluate how well an LLM writes cuBLASDx / cuFFTDx code, with and without help.

Conditions (see eval/prompts.py):
    baseline          task description only
    context           + context pack (guide, pitfalls, API index generated from the headers)
    context_feedback  + context pack + up to --max-repairs rounds of compiler/test feedback

Every generated file is compiled against the task harness, run on the GPU and checked
against a double-precision CPU reference. A sample PASSES if it compiles, runs, is
numerically correct AND really calls the Dx library inside a kernel.

Examples:
    python eval/run_eval.py --oracle                         # validate harnesses, reference timings
    export ANTHROPIC_API_KEY=...
    python eval/run_eval.py --model claude-haiku-5-5 --samples 3
    python eval/run_eval.py --model claude-sonnet-5-5 --tasks b1_gemm_batched_32,f1_fft_c2c_256
    python eval/report.py results/*.jsonl
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)
import dxtools  # noqa: E402
import prompts  # noqa: E402

RESULTS = os.path.join(dxtools.ROOT, "results")


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
def make_llm(provider, model, max_tokens):
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()

        def call(system, messages):
            r = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.usage.input_tokens, r.usage.output_tokens
        return call

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()

        def call(system, messages):
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}] + messages)
            u = r.usage
            return r.choices[0].message.content, u.prompt_tokens, u.completion_tokens
        return call

    if provider == "mock":  # pipeline test without API: reference on first try, broken otherwise
        def call(system, messages):
            task = re.search(r"# (\w+) ", messages[0]["content"]).group(1)
            task_dir = next(t for t in dxtools.list_tasks() if t.startswith(task))
            with open(os.path.join(dxtools.TASKS_DIR, task_dir, "reference.cu")) as f:
                code = f.read()
            if len(messages) == 1 and "sample=1" in system:
                code = code.replace("+ cublasdx::Block()", "").replace("+ cufftdx::Block()", "")
            return f"```cuda\n{code}\n```", 0, 0
        return call

    raise ValueError(f"unknown provider {provider}")


def with_retries(fn, *args, tries=4):
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as e:  # rate limits, transient network errors
            if i == tries - 1:
                raise
            wait = 10 * (i + 1)
            print(f"  LLM call failed ({type(e).__name__}: {str(e)[:120]}), retry in {wait}s", flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def uses_library(task, code):
    lib = prompts.library_of(task)
    return bool(re.search(rf"\b{lib}::", code)) and re.search(r"\.execute\s*\(", code) is not None


COMPILE_CATEGORIES = [
    # traits queried on a description without Block()/Thread(): "fft_description<...> has no member"
    ("missing_execution_operator", r"detail::(fft|blas|solver)_description<[^\n]*has no member"),
    ("incomplete_description", r"must be complete|description is not complete|incomplete"),
    ("wrong_api_name", r"has no member|is not a member|identifier \"[^\"]+\" is undefined|is not a template|"
                       r"no template named|namespace \"\w+\" has no member"),
    ("type_mismatch", r"no instance of|argument of type|cannot be used to initialize|incompatible|"
                      r"no suitable conversion|a value of type"),
    ("static_assert", r"static assertion failed"),
    ("syntax", r"expected a|expected an|unexpected"),
]


def compile_category(log):
    for name, pat in COMPILE_CATEGORIES:
        if re.search(pat, log):
            return "compile:" + name
    return "compile:other"


# ---------------------------------------------------------------------------
# One sample = one conversation (1 attempt, or several with feedback)
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, args, llm, oracle):
        self.a = args
        self.llm = llm
        self.oracle = oracle
        self.compile_sem = threading.Semaphore(args.compile_jobs)
        self.gpu_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.code_dir = os.path.join(RESULTS, args.run_name, "code")
        os.makedirs(self.code_dir, exist_ok=True)

    def test(self, task, code, tag):
        """compile + run one candidate. Returns an attempt record."""
        src = os.path.join(self.code_dir, f"{tag}.cu")
        with open(src, "w") as f:
            f.write(code)
        rec = {"uses_library": uses_library(task, code)}
        with tempfile.TemporaryDirectory() as d:
            binary = os.path.join(d, "bench")
            with self.compile_sem:
                ok, log = dxtools.compile_task(task, src, self.a.arch, binary)
            if not ok:
                rec.update(stage="compile", category=compile_category(log), log=log)
                return rec
            if self.a.compile_only:
                rec.update(stage="compiled", category="compiled_only", log="")
                return rec
            with self.gpu_lock:
                res = dxtools.run_binary(binary, timeout=self.a.run_timeout)
        rec["result"] = res
        if res.get("status") != "ok":
            rec.update(stage="runtime", category="runtime:" + res.get("status", "?"), log=json.dumps(res))
        elif not res.get("correct"):
            rec.update(stage="wrong_result", category="wrong_result", log=json.dumps(res))
        elif not rec["uses_library"]:
            rec.update(stage="no_library", category="no_library", log="")
        else:
            rec.update(stage="pass", category="pass", log="")
        return rec

    def sample(self, task, condition, k):
        system = prompts.system_prompt(task, condition)
        if self.a.provider == "mock":
            system += f"\nsample={k}"
        messages = [{"role": "user", "content": prompts.task_prompt(task)}]
        max_attempts = 1 + (self.a.max_repairs if condition == "context_feedback" else 0)
        attempts, tok_in, tok_out = [], 0, 0
        t0 = time.time()
        for j in range(max_attempts):
            text, ti, to = with_retries(self.llm, system, messages)
            tok_in += ti
            tok_out += to
            code = dxtools.extract_code(text)
            rec = self.test(task, code, f"{task}__{condition}__s{k}__a{j}")
            attempts.append({key: rec.get(key) for key in ("stage", "category", "uses_library", "result")}
                            | {"log": (rec.get("log") or "")[:1500]})
            if rec["stage"] in ("pass", "compiled"):
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": prompts.feedback_prompt(rec["stage"], rec.get("log", "")[:5000])}]
        last = attempts[-1]
        res = last.get("result") or {}
        ref_ms = (self.oracle.get(task) or {}).get("time_ms")
        out = {
            "run": self.a.run_name, "model": self.a.model, "provider": self.a.provider, "task": task,
            "library": prompts.library_of(task), "condition": condition, "sample": k,
            "passed": last["stage"] == "pass", "compiled": last["stage"] not in ("compile",),
            "first_try_passed": attempts[0]["stage"] == "pass", "attempts_used": len(attempts),
            "final_category": last["category"], "first_category": attempts[0]["category"],
            "rel_err": res.get("rel_err"), "time_ms": res.get("time_ms"), "ref_time_ms": ref_ms,
            "speed_vs_ref": (ref_ms / res["time_ms"]) if (ref_ms and res.get("time_ms") and last["stage"] == "pass") else None,
            "tokens_in": tok_in, "tokens_out": tok_out, "seconds": round(time.time() - t0, 1),
            "attempts": attempts,
        }
        with self.write_lock:
            with open(self.a.out, "a") as f:
                f.write(json.dumps(out) + "\n")
        mark = "PASS" if out["passed"] else out["final_category"]
        print(f"[{condition:16s}] {task:34s} s{k} -> {mark} ({len(attempts)} attempt(s), {out['seconds']}s)",
              flush=True)
        return out


# ---------------------------------------------------------------------------
def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       text=True, timeout=20).strip().splitlines()[0]
    except Exception:
        return "unknown"


def run_oracle(args):
    """Compile and run every reference solution: validates harnesses + gives reference timings."""
    out, fails = {}, 0
    for task in args.tasks:
        with tempfile.TemporaryDirectory() as d:
            binary = os.path.join(d, "bench")
            ok, log = dxtools.compile_task(task, os.path.join(dxtools.TASKS_DIR, task, "reference.cu"),
                                           args.arch, binary)
            if not ok:
                print(f"FAIL {task}: reference does not compile\n{log[:2000]}")
                fails += 1
                continue
            if args.compile_only:
                print(f"OK   {task} (compiled)")
                continue
            res = dxtools.run_binary(binary, timeout=args.run_timeout)
        good = res.get("status") == "ok" and res.get("correct")
        fails += not good
        print(f"{'OK  ' if good else 'FAIL'} {task:34s} rel_err={res.get('rel_err')} "
              f"time={res.get('time_ms')} ms  {res.get('message', '')}")
        out[task] = res | {"gpu": gpu_name(), "arch": args.arch}
    if out:
        path = os.path.join(RESULTS, "oracle.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print("reference timings written to", path)
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "mock"])
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--conditions", default="baseline,context,context_feedback")
    p.add_argument("--tasks", default="all", help="comma-separated task ids (or prefixes), default all")
    p.add_argument("--samples", type=int, default=3, help="independent samples per task and condition")
    p.add_argument("--max-repairs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--arch", default=None, help="sm arch without dot, default: detected GPU")
    p.add_argument("--workers", type=int, default=4, help="concurrent conversations")
    p.add_argument("--compile-jobs", type=int, default=2, help="concurrent nvcc processes")
    p.add_argument("--run-timeout", type=int, default=120)
    p.add_argument("--compile-only", action="store_true", help="no GPU: stop after compilation")
    p.add_argument("--oracle", action="store_true", help="run the reference solutions only")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    args.arch = args.arch or dxtools.detect_arch()
    all_tasks = dxtools.list_tasks()
    if args.tasks == "all":
        args.tasks = all_tasks
    else:
        wanted = args.tasks.split(",")
        args.tasks = [t for t in all_tasks if any(t.startswith(w) for w in wanted)]
    os.makedirs(RESULTS, exist_ok=True)
    print(f"arch sm_{args.arch} | {len(args.tasks)} tasks | nvcc: {dxtools.find_nvcc()} | MathDx: {dxtools.find_mathdx()}")

    if args.oracle:
        sys.exit(run_oracle(args))

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    safe_model = re.sub(r"[^\w.-]", "_", args.model)
    args.run_name = f"{safe_model}_{stamp}"
    args.out = args.out or os.path.join(RESULTS, f"{args.run_name}.jsonl")
    oracle = {}
    if os.path.exists(os.path.join(RESULTS, "oracle.json")):
        with open(os.path.join(RESULTS, "oracle.json")) as f:
            oracle = json.load(f)
    else:
        print("note: results/oracle.json missing -> no speed comparison (run with --oracle first)")

    llm = make_llm(args.provider, args.model, args.max_tokens)
    ev = Evaluator(args, llm, oracle)
    jobs = [(t, c, k) for c in args.conditions.split(",") for t in args.tasks for k in range(args.samples)]
    print(f"{len(jobs)} samples -> {args.out}")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ev.sample, *j) for j in jobs]
        for fut in cf.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("sample crashed:", repr(e)[:300], flush=True)
    print(f"\ndone. Summarise with:  python eval/report.py {args.out}")


if __name__ == "__main__":
    main()
