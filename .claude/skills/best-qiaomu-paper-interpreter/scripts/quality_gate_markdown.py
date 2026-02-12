#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quality gate for diagram-heavy markdown.

Goal: fail closed on diagrams that are likely to be misleading / unreadable:
- Mermaid blocks: basic structural checks + long-label wrapping enforcement
- Canvas: disallow (require static images instead)
- Excalidraw: disallow direct embeds/links in final reading markdown
- Images: verify local paths exist (best-effort)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _iter_mermaid_blocks(md: str):
    # ```mermaid ... ```
    fence = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    for m in fence.finditer(md):
        yield m.start(), m.group(1).strip("\n")


def _label_wrap_ok(label: str) -> bool:
    s = label.strip()
    if len(s) <= 24:
        return True
    # If it's long, require explicit wrapping.
    return ("<br" in s) or ("\\n" in s) or ("\n" in s)


def _check_mermaid(code: str) -> list[str]:
    issues: list[str] = []
    lines = [ln.rstrip() for ln in code.splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return ["empty mermaid block"]

    first = nonempty[0].strip()
    if not (
        first.startswith("flowchart")
        or first.startswith("graph")
        or first.startswith("sequenceDiagram")
        or first.startswith("stateDiagram")
        or first.startswith("classDiagram")
        or first.startswith("erDiagram")
        or first.startswith("mindmap")
    ):
        issues.append(f"first line should declare diagram type (got: {first[:48]}...)")

    if len(code) > 20000:
        issues.append("diagram too large (>20k chars)")
    if len(nonempty) > 250:
        issues.append("diagram too long (>250 non-empty lines)")

    # Mermaid commonly chokes on markdown list-looking labels.
    if re.search(r"\[[ \t]*\d+[.．、]", code):
        issues.append("node label looks like a markdown list (e.g., [1. ...])")

    # Too much parallel chaining makes layout unreadable.
    if re.search(r"-->\s*[^\\n]*&[^\\n]*&", code):
        issues.append("too many parallel edges in one line (A --> B & C & D ...)")

    # Node labels: enforce wrap for long labels in common shapes.
    for m in re.finditer(r"\[([^\[\]]+)\]", code):
        lab = m.group(1)
        if not _label_wrap_ok(lab):
            issues.append(f"long node label not wrapped: [{lab[:48]}...]")
            break

    for m in re.finditer(r"\(([^\(\)]+)\)", code):
        lab = m.group(1)
        if len(lab.strip()) > 40 and not _label_wrap_ok(lab):
            issues.append(f"long node label not wrapped: ({lab[:48]}...)")
            break

    for m in re.finditer(r"\{([^\{\}]+)\}", code):
        lab = m.group(1)
        if len(lab.strip()) > 40 and not _label_wrap_ok(lab):
            issues.append(f"long node label not wrapped: {{{lab[:48]}...}}")
            break

    # Heuristic node count (best-effort): ID + shape.
    node_hits = re.findall(r"(^|[ \t])([A-Za-z][A-Za-z0-9_]*)\s*[\[\(\{]", code, flags=re.MULTILINE)
    if len(node_hits) > 35:
        issues.append(f"too many nodes (~{len(node_hits)}) - likely unreadable")

    return issues


def _iter_local_images(md: str):
    # Markdown images: ![alt](path)
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
        yield m.group(1).strip()
    # HTML images: <img src="...">
    for m in re.finditer(r"<img[^>]+src=['\"]([^'\"]+)['\"][^>]*>", md, flags=re.IGNORECASE):
        yield m.group(1).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown_path", type=str)
    args = ap.parse_args()

    md_path = Path(args.markdown_path).expanduser().resolve()
    if not md_path.exists():
        print(f"ERROR: markdown not found: {md_path}", file=sys.stderr)
        return 2

    md = md_path.read_text(encoding="utf-8", errors="replace")
    base_dir = md_path.parent

    errors: list[str] = []
    warnings: list[str] = []

    if re.search(r"<canvas\b", md, flags=re.IGNORECASE):
        errors.append("canvas tag found in markdown: require static PNG instead of <canvas>")

    if re.search(r"\.excalidraw\.md\b", md, flags=re.IGNORECASE):
        errors.append("excalidraw source file referenced in final markdown: require exported PNG instead")

    # Prevent tool/patch artifacts from leaking into final reading output.
    if re.search(r"\*\*\*\s*End Patch|\*\*\*\s*Begin Patch|recipient_name|apply_patch", md, flags=re.IGNORECASE):
        errors.append("patch/tool artifact found in markdown (e.g. *** End Patch / recipient_name)")

    # Mermaid checks
    mermaid_blocks = list(_iter_mermaid_blocks(md))
    for idx, (pos, code) in enumerate(mermaid_blocks, start=1):
        issues = _check_mermaid(code)
        for it in issues:
            errors.append(f"Mermaid#{idx}: {it}")

    # Local image existence
    for img in _iter_local_images(md):
        # Skip remote/data.
        if img.startswith("http://") or img.startswith("https://") or img.startswith("data:"):
            continue
        # Strip optional title: path "title"
        img2 = img.split()[0]
        p = Path(img2)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            warnings.append(f"image missing: {img2} (resolved: {p})")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"- {w}")
        print("")

    if errors:
        print("FAILED quality gate:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("OK: quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
