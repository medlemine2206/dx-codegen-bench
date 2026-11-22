"""MCP server exposing the Dx context pack and a compiler to coding assistants.

Tools:
    list_docs()                 -> names of the context documents
    read_doc(name)              -> one document (cuBLASDx / cuFFTDx guide, pitfalls, API index)
    search_docs(query)          -> the most relevant sections across all documents
    compile_check(code, arch)   -> compiles a .cu snippet against MathDx, returns errors

Run (stdio transport, for Claude Desktop / Claude Code / any MCP client):
    pip install mcp nvidia-mathdx==25.6.0      (works with MCP SDK 1.x and 2.x)
    python tools/mcp_server.py

Claude Code:   claude mcp add dx-docs -- python /abs/path/tools/mcp_server.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import dxtools  # noqa: E402

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server  # noqa: E402
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # noqa: E402

mcp = _Server("dx-docs")


def _docs():
    out = {}
    for sub in ("", "generated"):
        d = os.path.join(dxtools.CONTEXT_DIR, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".md"):
                with open(os.path.join(d, fn)) as f:
                    out[os.path.join(sub, fn) if sub else fn] = f.read()
    return out


def _sections(name, text):
    parts = re.split(r"(?m)^(?=#{1,3} )", text)
    return [(name, p.strip()) for p in parts if p.strip()]


@mcp.tool()
def list_docs() -> list[str]:
    """List the available documents about NVIDIA MathDx device libraries (cuBLASDx, cuFFTDx)."""
    return list(_docs().keys())


@mcp.tool()
def read_doc(name: str) -> str:
    """Return a full document, e.g. 'cublasdx.md', 'cufftdx.md', 'pitfalls.md',
    'generated/cufftdx_api.md'."""
    docs = _docs()
    if name not in docs:
        return f"unknown document '{name}'. Available: {', '.join(docs)}"
    return docs[name]


@mcp.tool()
def search_docs(query: str, max_sections: int = 4) -> str:
    """Keyword search over all document sections; returns the best matching sections."""
    words = [w.lower() for w in re.findall(r"[A-Za-z_:<>]+", query) if len(w) > 2]
    scored = []
    for name, text in _docs().items():
        for doc, sec in _sections(name, text):
            low = sec.lower()
            score = sum(low.count(w) for w in words)
            if score:
                scored.append((score, doc, sec))
    scored.sort(key=lambda t: -t[0])
    if not scored:
        return "no match"
    return "\n\n---\n\n".join(f"[{doc}]\n{sec}" for _, doc, sec in scored[:max_sections])


@mcp.tool()
def compile_check(code: str, arch: str = "") -> str:
    """Compile a complete .cu file (that includes <cublasdx.hpp> / <cufftdx.hpp>) for the
    given architecture (e.g. '75' for T4, '80' for A100; default = local GPU). The macro
    BENCH_SM is defined as arch*10. Returns 'OK' or the compiler errors."""
    ok, log = dxtools.compile_check(code, arch or None)
    return "OK" if ok else log


if __name__ == "__main__":
    mcp.run()
