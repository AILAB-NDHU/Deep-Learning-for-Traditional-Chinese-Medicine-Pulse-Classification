"""
Reorder the manual `thebibliography` of paper_access.tex so that entries appear
in order of first citation (IEEE Access resubmission checklist, item
"list the references in order of citation").  Keys are untouched; only the
\\bibitem lines are permuted, so numbering follows citation order after
recompilation.  Uncited entries are reported and appended at the end.

Usage: python reorder_bib.py [--write]      (dry-run by default)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_access" / "paper_access.tex"


def main():
    src = TEX.read_text(encoding="utf-8")
    head, rest = src.split(r"\begin{thebibliography}", 1)
    bibopen, rest = rest.split("\n", 1)
    body, tail = rest.split(r"\end{thebibliography}", 1)

    items = re.findall(r"(\\bibitem\{([^}]+)\}.*?)(?=\n\\bibitem\{|\n*$)", body, flags=re.S)
    by_key = {k: txt.rstrip("\n") for txt, k in items}

    # order of first citation in the body text (before the bibliography), ignoring comments
    text = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in head.splitlines())   # keep \% (escaped)
    order: list[str] = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", text):
        for k in (s.strip() for s in m.group(1).split(",")):
            if k and k not in order:
                order.append(k)
    uncited = [k for k in by_key if k not in order]
    missing = [k for k in order if k not in by_key]

    print(f"{len(by_key)} bibitems, {len(order)} distinct cited keys")
    if missing:
        print("CITED BUT NOT IN BIBLIOGRAPHY:", missing)
    if uncited:
        print("IN BIBLIOGRAPHY BUT NEVER CITED (appended at end):", uncited)
    old_seq = [k for _, k in items]
    new_seq = [k for k in order if k in by_key] + uncited
    moved = sum(a != b for a, b in zip(old_seq, new_seq))
    print(f"positions that change: {moved}")
    for i, k in enumerate(new_seq, 1):
        j = old_seq.index(k) + 1
        flag = "" if i == j else f"   (was [{j}])"
        print(f"  [{i:2d}] {k}{flag}")

    if "--write" in sys.argv:
        new_body = "\n".join(by_key[k] for k in new_seq) + "\n"
        TEX.write_text(head + r"\begin{thebibliography}" + bibopen + "\n" + new_body
                       + r"\end{thebibliography}" + tail, encoding="utf-8")
        print(f"written: {TEX}")
    else:
        print("(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
