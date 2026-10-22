"""Which words cause the 465 zero-operator-slot MBPP docstrings?

Measures coverage, not opinion. For every docstring the operator channel is
empty (measured 0.4774 = 465/974), collect its tokens. Then simulate the
directive's proposed word lists and report how many gain a slot.

The point is to size the fix BEFORE editing the module, and to find words
the directive's lists MISS. A list copied without measurement would leave
the tail unaddressed and the rate above the 0.10 bar.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

import henri.operator_ingress as oi

REPO = Path(__file__).resolve().parents[1]

# Directly from the Phase 9 directive, minus the pure function words
# ("and", "or", "in", "not"). MEASURED REASON: "and" appears in both of the
# directive's own Step 2 fixtures, and mapping it would put a shared
# component in the operator slot, breaking the required cos == -1.0. These
# words are also near-universal in English prose, so mapping them would
# destroy discrimination corpus-wide rather than add information.
PROPOSED = {
    "ARITH_ADDITIVE": (+1, {"add", "added", "addition", "sum", "summation",
                            "total", "plus", "increment", "combine"}),
}
PROPOSED_PAIRS = {
    "ARITH_ADDITIVE": (
        {"add", "added", "addition", "sum", "summation", "total", "plus",
         "increment", "combine", "increases", "increase"},
        {"subtract", "subtraction", "difference", "minus", "decrement",
         "deduct", "less", "decreases", "decrease", "reduce", "remove"}),
    "ARITH_MULTIPLICATIVE": (
        {"multiply", "multiplied", "multiplication", "product", "times",
         "area", "volume", "double", "triple", "square", "cube"},
        {"divide", "divided", "division", "quotient", "ratio", "half",
         "average", "mean", "split"}),
    "EXTREMUM": (
        {"minimum", "min", "smallest", "lowest", "least", "shorter",
         "under", "below"},
        {"maximum", "max", "largest", "highest", "greatest", "longer",
         "over", "above"}),
    "PARITY": (
        {"even", "divisible", "multiple", "modulo", "remainder"},
        {"odd", "indivisible"}),
}

STOP = set("""a an the of to in on for with and or not is are be as by from
at it its this that these those given write function program accept return
returns string list tuple dict input output numbers number integer integers
positive negative value values element elements array sequence using use
which each every all any both one two three first second""".split())

TOK = re.compile(r"[a-z]+")


def main() -> int:
    rows = [json.loads(l) for l in
            (REPO / "data" / "official_benchmarks" / "mbpp.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)

    zero_texts, have = [], 0
    for r in rows:
        text = (r.get("text") or "").strip()
        feats, _ = oi.all_features(text)
        opf, _st = oi.split_features(feats)
        if opf:
            have += 1
        else:
            zero_texts.append(text)

    print("docstrings %d  with op slot %d  zero %d  zero_fraction %.4f"
          % (n, have, len(zero_texts), len(zero_texts) / float(n)))

    # Existing lexicon keys, for the overlap question.
    existing = set(oi.LEXICON)
    print("existing LEXICON keys %d" % len(existing))

    # Token frequency inside the zero-slot docstrings only.
    cnt = Counter()
    for t in zero_texts:
        for w in TOK.findall(t.lower()):
            if w not in STOP and len(w) > 2:
                cnt[w] += 1
    print("\ntop 40 content words in ZERO-SLOT docstrings:")
    for w, c in cnt.most_common(40):
        mark = "LEX" if w in existing else "   "
        prop = "PROP" if any(w in plus or w in minus
                             for plus, minus in PROPOSED_PAIRS.values()) else "    "
        print("   %-16s %4d  %s %s" % (w, c, mark, prop))

    # Simulate: which zero-slot docstrings would gain a slot under the
    # proposed lists? This is coverage arithmetic, not a claim about the
    # final numbers; the real rate is re-measured after the edit.
    all_prop = set()
    for plus, minus in PROPOSED_PAIRS.values():
        all_prop |= plus | minus
    gain = 0
    still = []
    for t in zero_texts:
        toks = set(TOK.findall(t.lower()))
        if toks & all_prop:
            gain += 1
        else:
            still.append(t)
    print("\nSIMULATION with proposed lists (function words excluded):")
    print("  gain a slot      %d" % gain)
    print("  still zero       %d" % len(still))
    print("  projected rate   %.4f   [bar <= 0.10]"
          % (len(still) / float(n)))

    print("\n20 docstrings that would STILL be zero:")
    for t in still[:20]:
        print("   - %s" % t[:96].replace("\n", " "))

    cnt2 = Counter()
    for t in still:
        for w in TOK.findall(t.lower()):
            if w not in STOP and len(w) > 2:
                cnt2[w] += 1
    print("\ntop 30 words in the STILL-ZERO residue:")
    print("   " + ", ".join("%s(%d)" % (w, c) for w, c in cnt2.most_common(30)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
