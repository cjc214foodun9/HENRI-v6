"""Measure the Step 2 fixture claims against the CURRENT module. No edits.

Three questions, all measured rather than assumed:

  M1  Does the module already produce a nonzero operator slot for prose?
  M2  What is cos(sum_doc, difference_doc)? The directive requires exactly
      -1.0. If shared words (notably "and") enter the operator slot, the
      algebra gives (|c|^2-|a|^2)/(|a|^2+|c|^2), which is 0.0 at equal
      weights. Measure, do not argue.
  M3  What fraction of MBPP docstrings has an identically zero operator
      slot? The directive cites 45% from Phase 8. Re-derive it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from henri import operator_ingress as oi

REPO = Path(__file__).resolve().parents[1]
SLOT = 16384

SUM_DOC = "Write a function to find the sum of a and b"
DIFF_DOC = "Write a function to find the difference of a and b"
ADD_CODE = "def f(a, b): return a + b"
SUB_CODE = "def f(a, b): return a - b"


def op_slot(text):
    """Operator-slot wave for one text, via the module's own split."""
    feats, _ = oi.all_features(text)
    opf, _struct = oi.split_features(feats)
    if not opf:
        return np.zeros(SLOT, dtype=np.complex128)
    return oi.ring_slot(opf, SLOT)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.real(np.vdot(a, b)) / (na * nb))


def main():
    print("=== M1/M2: fixture cosines, current module ===")
    for label, t in (("sum_doc", SUM_DOC), ("diff_doc", DIFF_DOC),
                     ("add_code", ADD_CODE), ("sub_code", SUB_CODE)):
        feats, info = oi.all_features(t)
        opf, struct = oi.split_features(feats)
        # MEASURED: `Feature` is `Tuple[str, float, int]`, not a dataclass or
        # namedtuple. `f.name` raised AttributeError. Index instead.
        names = [f[0] for f in opf][:8]
        print("  %-9s n_op=%-2d n_struct=%-2d norm=%.6f  %s"
              % (label, len(opf), len(struct),
                 float(np.linalg.norm(op_slot(t))), names))

    s, d = op_slot(SUM_DOC), op_slot(DIFF_DOC)
    a, b = op_slot(ADD_CODE), op_slot(SUB_CODE)
    print("\n  cos(sum_doc, diff_doc)      = %+0.12f   [directive: == -1.0]"
          % cos(s, d))
    print("  cos(sum_doc, add_code)      = %+0.12f   [directive: > +0.70]"
          % cos(s, a))
    print("  cos(sum_doc, sub_code)      = %+0.12f   [directive: < -0.70]"
          % cos(s, b))
    print("  cos(add_code, sub_code)     = %+0.12f   [Phase 8: -1.0]"
          % cos(a, b))

    # Decompose WHY. Which features are shared by both docstrings?
    fs, _ = oi.split_features(oi.all_features(SUM_DOC)[0])
    fd, _ = oi.split_features(oi.all_features(DIFF_DOC)[0])
    ns = {f[0] for f in fs}
    nd = {f[0] for f in fd}
    shared = sorted(ns & nd)
    print("\n  shared operator features: %s" % (shared or "NONE"))
    print("  sum-only: %s" % sorted(ns - nd))
    print("  diff-only: %s" % sorted(nd - ns))

    print("\n=== M3: MBPP zero-operator-slot fraction ===")
    rows = [json.loads(l) for l in
            (REPO / "data" / "official_benchmarks" / "mbpp.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    zero = 0
    for r in rows:
        feats, _ = oi.all_features((r.get("text") or "").strip())
        opf, _st = oi.split_features(feats)
        if not opf:
            zero += 1
    n = len(rows)
    print("  docstrings %d   zero operator slot %d   fraction %.4f"
          % (n, zero, zero / float(n)))
    print("  directive claims 0.4500 from Phase 8; bar after Step 2 <= 0.10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
