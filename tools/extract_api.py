"""Extract an LLM-friendly API index from the MathDx headers.

The Dx libraries are header-only C++ template libraries: their "API" is a set of
operator structs (Size<>, Precision<>, ...), enum values, traits exposed on the
description type (FFT::block_dim, BLAS::get_layout_smem_a(), ...) and a few free
functions. This script scans the installed headers and writes, for each library:

    context/generated/<lib>_api.json   machine-readable index
    context/generated/<lib>_api.md     compact markdown, meant to be pasted in a prompt

Because it reads the headers that are actually installed, the index always matches the
library version the code will be compiled against (no stale documentation).

    python tools/extract_api.py                       # auto-detect pip-installed MathDx
    python tools/extract_api.py --include /path/to/mathdx/include
"""
import argparse
import json
import os
import re
import sys

LIBS = {
    "cublasdx": {"exec_files": ["detail/blas_execution.hpp"],
                 "free_files": ["detail/copy.hpp", "detail/shared_memory.hpp", "detail/tensor.hpp"],
                 "version_file": "cublasdx_version.hpp", "version_macro": "CUBLASDX_VERSION"},
    "cufftdx": {"exec_files": ["detail/fft_execution.hpp"],
                "free_files": ["detail/workspace.hpp", "traits/fft_traits.hpp"],
                "version_file": "cufftdx_version.hpp", "version_macro": "CUFFTDX_VERSION"},
    "cusolverdx": {"exec_files": ["detail/solver_execution.hpp"],
                   "free_files": ["traits/solver_traits.hpp"],
                   "version_file": "cusolverdx_version.hpp", "version_macro": "CUSOLVERDX_VERSION"},
    "curanddx": {"exec_files": [], "free_files": [],
                 "version_file": "curanddx_version.hpp", "version_macro": "CURANDDX_VERSION"},
}


def find_include():
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        for base in spec.submodule_search_locations:
            inc = os.path.join(base, "mathdx", "include")
            if os.path.isdir(inc):
                return inc
    except Exception:
        pass
    return None


def strip_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def version(inc, lib, cfg):
    path = os.path.join(inc, lib, cfg["version_file"])
    if not os.path.exists(path):
        return None
    src = read(path)
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"#define\s+{cfg['version_macro']}_{part}\s+(\d+)", src)
        if m:
            parts.append(m.group(1))
    return ".".join(parts) if parts else None


def operators(inc, lib):
    """Public operator templates and enums declared in <lib>/operators/*.hpp."""
    ops, enums, aliases = [], {}, {}
    opdir = os.path.join(inc, lib, "operators")
    if not os.path.isdir(opdir):
        return ops, enums, aliases
    files = [os.path.join(opdir, fn) for fn in sorted(os.listdir(opdir)) if fn.endswith(".hpp")]
    top = os.path.join(inc, lib, "operators.hpp")  # Block, BlockDim, SM are declared here
    if os.path.exists(top):
        files.append(top)
    for path in files:
        fn = os.path.relpath(path, os.path.join(inc, lib))
        src = strip_comments(read(path))
        # keep only the public namespace part (drop commondx::detail specialisations)
        public = re.split(r"namespace\s+commondx::detail", src)[0]
        for m in re.finditer(r"template\s*<([^>]*(?:<[^>]*>[^>]*)*)>\s*struct\s+(\w+)\s*[:{]", public):
            params = " ".join(m.group(1).split())
            name = m.group(2)
            if name.startswith("is_") or name.startswith("get_") or name[0].islower():
                continue
            ops.append({"name": name, "template_params": params, "file": fn})
        for m in re.finditer(r"template\s*<([^>]*)>\s*using\s+(\w+)\s*=\s*commondx::", public):
            if m.group(2) not in [o["name"] for o in ops]:
                ops.append({"name": m.group(2), "template_params": " ".join(m.group(1).split()), "file": fn})
        for m in re.finditer(r"struct\s+(\w+)\s*:\s*public\s+commondx::\w+\s*\{", public):
            if m.group(1)[0].isupper() and m.group(1) not in [o["name"] for o in ops]:
                ops.append({"name": m.group(1), "template_params": "", "file": fn})
        for m in re.finditer(r"enum\s+class\s+(\w+)\s*(?::\s*\w+)?\s*\{([^}]*)\}", public):
            vals = [v.split("=")[0].strip() for v in m.group(2).split(",") if v.strip()]
            enums[m.group(1)] = vals
        for m in re.finditer(r"inline\s+constexpr\s+auto\s+(\w+)\s*=\s*([\w:]+)\s*;", public):
            aliases[m.group(1)] = m.group(2)
    return ops, enums, aliases


def description_traits(inc, lib, files):
    """`static constexpr` members, type aliases and member functions of execution descriptions."""
    traits, types, funcs = {}, {}, set()
    for rel in files:
        path = os.path.join(inc, lib, rel)
        if not os.path.exists(path):
            continue
        src = strip_comments(read(path))
        for m in re.finditer(r"static\s+constexpr\s+([\w:<>, ]+?)\s+(\w+)\s*=", src):
            name = m.group(2)
            if name.startswith("this_") or name.startswith("is_valid") or name.startswith("valid_") \
                    or name.startswith("has_") or name.startswith("_") or name.endswith("_v"):
                continue
            traits.setdefault(name, " ".join(m.group(1).split()))
        for m in re.finditer(r"\busing\s+(\w+_type|value_type)\s*=", src):
            if not m.group(1).startswith("this_"):
                types.setdefault(m.group(1), True)
        for m in re.finditer(r"(?:CUBLASDX_DEVICE|CUBLASDX_HOST_DEVICE|__device__|__host__)[^;{(]*?\b(execute|get_\w+|suggest_\w+)\s*\(", src):
            funcs.add(m.group(1))
    return traits, sorted(types), sorted(funcs)


def free_functions(inc, lib, files):
    out = set()
    for rel in files:
        path = os.path.join(inc, lib, rel)
        if not os.path.exists(path):
            continue
        src = strip_comments(read(path))
        public = re.split(r"namespace\s+detail\b", src)
        body = src if len(public) == 1 else src
        for m in re.finditer(r"(?:CUBLASDX_DEVICE|CUBLASDX_HOST_DEVICE|inline|auto|void)\s+(?:auto\s+|void\s+|constexpr\s+)*"
                             r"(copy_wait|copy|make_tensor|slice_shared_memory\w*|get_shared_storage_size\w*|"
                             r"make_workspace|size_of|make_gmem_\w+|make_smem_\w+)\s*\(", body):
            out.add(m.group(1))
        for m in re.finditer(r"\b(size_of|type_of|direction_of|precision_of|sm_of|block_dim_of|"
                             r"leading_dimension_of|alignment_of|arrangement_of|function_of)\b\s*[<{]", body):
            out.add(m.group(1) + "<Description>")
    return sorted(out)


def to_markdown(lib, info):
    lines = [f"# {lib} {info['version'] or ''} — API index (generated from headers)", ""]
    lines.append("## Description operators (combine with `+` inside `decltype(...)`)")
    for op in info["operators"]:
        p = f"<{op['template_params']}>" if op["template_params"] else "()"
        lines.append(f"- `{lib}::{op['name']}{p}`")
    if info["enums"]:
        lines += ["", "## Enum values"]
        for name, vals in info["enums"].items():
            lines.append(f"- `{lib}::{name}`: " + ", ".join(f"`{v}`" for v in vals))
    if info["aliases"]:
        lines += ["", "## Short aliases"]
        lines.append(", ".join(f"`{lib}::{k}` = `{v}`" for k, v in sorted(info["aliases"].items())))
    if info["traits"]:
        lines += ["", "## Traits available on a complete description type (e.g. `X::name`)"]
        lines.append(", ".join(f"`{k}`" for k in sorted(info["traits"])))
    if info["types"]:
        lines += ["", "## Member types"]
        lines.append(", ".join(f"`{t}`" for t in info["types"]))
    if info["member_functions"]:
        lines += ["", "## Member functions"]
        lines.append(", ".join(f"`{f}(...)`" for f in info["member_functions"]))
    if info["free_functions"]:
        lines += ["", "## Free functions / trait templates"]
        lines.append(", ".join(f"`{lib}::{f}`" for f in info["free_functions"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default=None, help="MathDx include directory")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "context", "generated"))
    a = ap.parse_args()
    inc = a.include or find_include()
    if not inc or not os.path.isdir(inc):
        sys.exit("MathDx include directory not found (pip install nvidia-mathdx or pass --include)")
    os.makedirs(a.out, exist_ok=True)
    for lib, cfg in LIBS.items():
        if not os.path.isdir(os.path.join(inc, lib)):
            continue
        ops, enums, aliases = operators(inc, lib)
        traits, types, mfuncs = description_traits(inc, lib, cfg["exec_files"])
        info = {"library": lib, "version": version(inc, lib, cfg), "operators": ops, "enums": enums,
                "aliases": aliases, "traits": traits, "types": types, "member_functions": mfuncs,
                "free_functions": free_functions(inc, lib, cfg["free_files"])}
        with open(os.path.join(a.out, f"{lib}_api.json"), "w") as f:
            json.dump(info, f, indent=1)
        md = to_markdown(lib, info)
        with open(os.path.join(a.out, f"{lib}_api.md"), "w") as f:
            f.write(md)
        print(f"{lib:11s} v{info['version']}: {len(ops)} operators, {len(enums)} enums, "
              f"{len(traits)} traits, {len(md)} chars")


if __name__ == "__main__":
    main()
