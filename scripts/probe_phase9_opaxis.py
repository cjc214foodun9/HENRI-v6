"""Phase 9 Step 2 gate: cross-modal lexical-to-operator homomorphism.

PRE-REGISTERED CLAUSES, fixed before the first run of this file.

  S2-1  cosine(sum_doc, difference_doc) == -1.0     (exact, tol 1e-12)
  S2-2  cosine(sum_doc, "return a + b")  > +0.70
  S2-3  cosine(sum_doc, "return a - b")  < -0.70
  S2-4  zero operator slot, 974 MBPP docstrings  <= 0.10
  S2-5  BLEED CONTROL: cosine(sum_doc, "return a * b") must be < +0.70.
        Without this, a lexicon that maps every word to one root would
        pass S2-2/S2-3 by making all arithmetic look additive.
  S2-6  COLLISION CONTROL: mean pairwise operator-slot cosine over a
        fixed 300-docstring sample must be < 0.10. Adding near-universal
        words raises this. A channel that assigns every docstring the same
        direction discriminates nothing, however many slots it fills.
  S2-7  Phase 8 fixture preserved: cosine("return a + b", "return a - b")
        == -1.0 exactly.

S2-4 IS EXPECTED TO FAIL, and the failure is a measurement, not a
tuning problem. MEASURED: the directive's proposed word lists gain 58 of
the 465 zero-slot docstrings, projecting 0.4179. The residue is
"find/check/convert/sort/regex/extract" — docstrings that describe
string, list, and regex manipulation with no arithmetic operation at
all. For those, an empty operator slot is CORRECT. Driving the rate to
0.10 would require mapping semantically empty verbs into the operator
basis, which collapses S2-6. The clause is reported, not relaxed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from henri import operator_ingress as oi

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "receipts" / "phase9_opaxis.json"
SLOT = 16384

SUM_DOC = "Write a function to find the sum of a and b"
DIFF_DOC = "Write a function to find the difference of a and b"
ADD_CODE = "def f(a, b): return a + b"
SUB_CODE = "def f(a, b): return a - b"
MUL_CODE = "def f(a, b): return a * b"
DIV_CODE = "def f(a, b): return a / b"

N_SAMPLE = 300
SAMPLE_SEED = 20261021


def op_slot(text: str) -> np.ndarray:
    feats, _info = oi.all_features(text)
    opf, _struct = oi.split_features(feats)
    if not opf:
        return np.zeros(SLOT, dtype=np.complex128)
    return oi.ring_slot(opf, SLOT)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.real(np.vdot(a, b)) / (na * nb))


def main() -> int:
    s, d = op_slot(SUM_DOC), op_slot(DIFF_DOC)
    a, b = op_slot(ADD_CODE), op_slot(SUB_CODE)
    m, v = op_slot(MUL_CODE), op_slot(DIV_CODE)

    c1 = cos(s, d)
    c2 = cos(s, a)
    c3 = cos(s, b)
    c5 = cos(s, m)
    c7 = cos(a, b)
    c_bleed_div = cos(s, v)

    rows = [json.loads(line) for line in
            (REPO / "data" / "official_benchmarks" / "mbpp.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(rows)
    zero = 0
    for r in rows:
        feats, _i = oi.all_features((r.get("text") or "").strip())
        opf, _s = oi.split_features(feats)
        if not opf:
            zero += 1
    rate = zero / float(n)

    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(n, size=min(N_SAMPLE, n), replace=False)
    vecs = [op_slot((rows[int(i)].get("text") or "").strip())
            for i in idx]
    nz = [x for x in vecs if float(np.linalg.norm(x)) > 1e-30]
    pair = []
    for i in range(len(nz)):
        for j in range(i + 1, len(nz)):
            pair.append(cos(nz[i], nz[j]))
    mean_pair = float(np.mean(pair)) if pair else 0.0
    frac_positive = float(np.mean([1.0 if p > 0.0 else 0.0 for p in pair])) \
        if pair else 0.0

    clauses = [
        {"id": "S2-1", "what": "cos(sum_doc, diff_doc) == -1.0",
         "value": round(c1, 15), "bar": "== -1.0 (tol 1e-12)",
         "pass": abs(c1 + 1.0) <= 1e-12},
        {"id": "S2-2", "what": "cos(sum_doc, code +) > +0.70",
         "value": round(c2, 15), "bar": "> +0.70", "pass": c2 > 0.70},
        {"id": "S2-3", "what": "cos(sum_doc, code -) < -0.70",
         "value": round(c3, 15), "bar": "< -0.70", "pass": c3 < -0.70},
        {"id": "S2-4", "what": "zero operator slot fraction <= 0.10",
         "value": round(rate, 6), "bar": "<= 0.10",
         "pass": rate <= 0.10, "zero": zero, "n": n},
        {"id": "S2-5", "what": "bleed control: cos(sum_doc, code *) < +0.70",
         "value": round(c5, 15), "bar": "< +0.70", "pass": c5 < 0.70},
        {"id": "S2-6", "what": "mean pairwise op-slot cosine < 0.10",
         "value": round(mean_pair, 9), "bar": "< 0.10",
         "pass": mean_pair < 0.10, "pairs": len(pair),
         "frac_positive_pairs": round(frac_positive, 6)},
        {"id": "S2-7", "what": "cos(code +, code -) == -1.0 (Phase 8 held)",
         "value": round(c7, 15), "bar": "== -1.0 (tol 1e-12)",
         "pass": abs(c7 + 1.0) <= 1e-12},
    ]

    npass = sum(1 for c in clauses if c["pass"])
    verdict = ("S2_PASS" if all(c["pass"] for c in clauses)
               else "S2_PARTIAL_%d_OF_%d" % (npass, len(clauses)))

    body = {
        "probe": "phase9_step2_cross_modal_operator",
        "slot_dim": SLOT,
        "cos_bleed_div": round(c_bleed_div, 15),
        "clauses": clauses,
        "n_clauses_pass": npass,
        "verdict": verdict,
        "mechanism_note": (
            "S2-4 fails by design. The directive's own word lists gain 58 "
            "of 465 zero-slot docstrings. The residue is find/check/"
            "convert/sort/regex/extract, which name no arithmetic "
            "operation. An empty operator slot is correct for them. "
            "Forcing the rate to 0.10 requires mapping semantically empty "
            "verbs, which collapses the S2-6 collision control."),
        "lexicon_size": len(oi.LEXICON),
    }

    ser = json.dumps(body, indent=2, sort_keys=True, default=float)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(ser.encode("utf-8"))
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    body["receipt_sha256"] = digest
    OUT.write_bytes(json.dumps(body, indent=2, sort_keys=True,
                               default=float).encode("utf-8"))

    for c in clauses:
        print("%-6s %-52s %+0.15g  %-22s %s"
              % (c["id"], c["what"], c["value"], c["bar"],
                 "PASS" if c["pass"] else "FAIL"))
    print("\nlexicon keys %d   zero-slot %d/%d = %.4f"
          % (len(oi.LEXICON), zero, n, rate))
    print("coverage gain vs 0.4774 baseline: %.4f -> %.4f (%+0.4f)"
          % (0.4774, rate, rate - 0.4774))
    print("receipt %s sha256 %s" % (OUT.name, digest[:16]))
    print("VERDICT %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
