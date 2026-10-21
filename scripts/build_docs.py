"""Render docs/project_description.pdf and docs/code_documentation.pdf with headless Chrome.

Requires: pip install playwright (uses the system Google Chrome via channel="chrome").
"""
from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SOURCES = ["app.py", "views/examine.py", "views/reliability.py", "views/provenance.py",
           "xraylens/model.py", "xraylens/explainability.py", "xraylens/inputs.py",
           "xraylens/perturbations.py", "xraylens/evaluation.py", "xraylens/app_state.py", "xraylens/ui.py",
           "scripts/fetch_eval_data.py", "scripts/run_evaluation.py", "scripts/build_gallery.py",
           "scripts/export_onnx.py", "scripts/export_web.py", "scripts/check_web_parity.py", "scripts/serve_web.py",
           "web/js/model.js", "web/js/imaging.js", "web/js/examine.js", "web/js/reliability.js", "web/js/common.js",
           "web/index.html", "web/vercel.json",
           "tests/conftest.py", "tests/test_preprocessing.py", "tests/test_inference.py",
           "requirements.txt", ".streamlit/config.toml"]

PY_TOKEN = re.compile(
    r'(?P<str>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|f?"(?:[^"\\\n]|\\.)*"|f?\'(?:[^\'\\\n]|\\.)*\')'
    r'|(?P<com>#[^\n]*)'
    r'|(?P<kw>\b(?:def|class|return|import|from|as|if|elif|else|for|while|in|not|and|or|with|try|except|finally|'
    r'raise|lambda|yield|None|True|False|is|pass|break|continue|global|assert)\b)')


def highlight(code: str) -> str:
    out, pos = [], 0
    for m in PY_TOKEN.finditer(code):
        out.append(html.escape(code[pos:m.start()]))
        cls = m.lastgroup
        out.append(f'<span class="{cls}">{html.escape(m.group())}</span>')
        pos = m.end()
    out.append(html.escape(code[pos:]))
    return "".join(out)


def code_html() -> str:
    built = date.today().isoformat()
    toc, body = [], []
    total = 0
    for i, rel in enumerate(SOURCES, 1):
        text = (ROOT / rel).read_text()
        n = text.count("\n") + 1
        total += n
        toc.append(f"<tr><td>{i:02d}</td><td>{rel}</td><td>{n}</td></tr>")
        src = highlight(text) if rel.endswith((".py", ".js")) else html.escape(text)
        lines = src.split("\n")
        numbered = "\n".join(f'<span class="ln">{k:>4}</span>{line}' for k, line in enumerate(lines, 1))
        body.append(f'<section><h2><span>{i:02d}</span> {rel}</h2><pre>{numbered}</pre></section>')
    readme = (ROOT / "README.md").read_text()
    arch = readme.split("## How it works", 1)[1].split("### Design decisions", 1)[0]
    arch_pre = html.escape(arch.split("```")[1].strip())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>XRayLens AI: Code Documentation</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
@page {{ size: Letter; margin: .55in .55in .6in; }}
body {{ font: 10pt/1.45 'Newsreader', Georgia, serif; color: #14181b; }}
.kick {{ font: 600 7pt 'JetBrains Mono', monospace; letter-spacing: .24em; text-transform: uppercase; color: #b86e00; }}
h1 {{ font: 400 44pt/.9 'Instrument Serif', serif; margin: .1in 0 .12in; }} h1 em {{ color: #b86e00; }}
h2 {{ font: 400 16pt 'Instrument Serif', serif; border-bottom: 1pt solid #14181b; padding-bottom: .04in; margin: 0 0 .08in; }}
h2 span {{ font-style: italic; color: #b86e00; margin-right: .08in; }}
section {{ break-before: page; }}
pre {{ font: 7.3pt/1.38 'JetBrains Mono', monospace; white-space: pre-wrap; word-break: break-word; margin: 0; }}
.ln {{ color: #9aa3a6; user-select: none; margin-right: .12in; }}
.kw {{ color: #8a3b9a; font-weight: 600; }} .str {{ color: #1f7f72; }} .com {{ color: #8a8f91; font-style: italic; }}
table {{ border-collapse: collapse; font: 8.5pt 'JetBrains Mono', monospace; width: 100%; }}
td, th {{ border-bottom: .5pt solid #14181b2e; padding: .04in .1in .04in 0; text-align: left; }}
.arch {{ background: #f3efe5; padding: .12in; border-left: 2.4pt solid #b86e00; }}
</style></head><body>
<div class="kick">Code documentation · built {built} · {len(SOURCES)} files · {total:,} lines</div>
<h1>XRay<em>Lens</em> AI: source code</h1>
<p>This document contains every file written for XRayLens AI. The model weights (Purino/resnet50-chest-xray-pneumonia) and
the evaluation datasets are third-party and downloaded at runtime by the scripts below. Run order:
<code>pip install -r requirements.txt</code> → <code>streamlit run app.py</code>. To reproduce the evaluation:
<code>scripts/fetch_eval_data.py</code> → <code>scripts/run_evaluation.py</code> → <code>scripts/build_gallery.py</code> → <code>pytest</code>.</p>
<h2 style="margin-top:.2in"><span>§</span> Architecture</h2><pre class="arch">{arch_pre}</pre>
<h2 style="margin-top:.2in"><span>§</span> Contents</h2><table><tr><th>#</th><th>File</th><th>Lines</th></tr>{''.join(toc)}</table>
{''.join(body)}
</body></html>"""


def main():
    (DOCS / "code_documentation.html").write_text(code_html())
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        for name in ("project_description", "code_documentation"):
            page.goto((DOCS / f"{name}.html").as_uri())
            page.wait_for_load_state("networkidle")
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(800)
            opts = {"path": str(DOCS / f"{name}.pdf"), "print_background": True, "prefer_css_page_size": True}
            if name == "code_documentation":
                opts["display_header_footer"] = True
                opts["header_template"] = "<span></span>"
                opts["footer_template"] = ('<div style="font:7px monospace;color:#888;width:100%;text-align:center">'
                                           'XRayLens AI · <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
            page.pdf(**opts)
            print("wrote", DOCS / f"{name}.pdf")
        browser.close()
    (DOCS / "code_documentation.html").unlink()


if __name__ == "__main__":
    main()
