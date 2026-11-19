"""Stateful multi-statement schemas for cold-start synthesis (Cycle AA2).

Z2's cold synthesizer emitted single-expression bodies.  HumanEval needs
state: an accumulator, a scan with a pointer pair, a grouping dictionary, a
character buffer, or a bounded recurrence.  This module builds those sketches
from the parsed ProblemSpec only: no task name, no task id, no benchmark
function name enters any branch.  A schema is instantiated with the parameter
NAMES the signature declares and the literal constants the public examples
carry, and every candidate is verified by the same sandbox check.

Families:
  (a) filtered sequence / index accumulation
  (b) two-pointer, sliding window and adjacent-pair scan
  (c) dictionary / frequency grouping and reduction
  (d) stateful string and character buffer transduction
  (e) bounded arithmetic and modular recurrence

The candidates are ranked by the caller from their inferred output kind, the
shape facts read off the public examples, and their keyword tags.  Nothing
here decides that a candidate is correct: the sandbox does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from henri.holarch.cold_synth import ProblemSpec


def _names(spec: ProblemSpec, kinds: Sequence[str]) -> List[str]:
    out: List[str] = []
    for param in spec.params:
        name = str(param["name"])
        kind = str(spec.arg_kinds.get(name, "any"))
        if kind in kinds:
            out.append(name)
    return out


def _constants(spec: ProblemSpec, limit: int = 4) -> List[int]:
    out: List[int] = []
    for row in spec.examples:
        for value in list(row.get("args", [])) + [row.get("expected")]:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 0 <= int(value) <= 32 and int(value) not in out:
                out.append(int(value))
        for value in list(row.get("args", [])):
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, int) and not isinstance(item, bool):
                        if 0 <= int(item) <= 32 and int(item) not in out:
                            out.append(int(item))
    return out[:limit]


def stateful_candidates(spec: ProblemSpec) -> List[Dict[str, Any]]:
    """Every stateful sketch the signature and the constants admit."""
    seqs = _names(spec, ("list", "tuple", "set"))
    strs = _names(spec, ("str",))
    nums = _names(spec, ("int", "float"))
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def add(tags: Sequence[str], body: str, effect: str, needs: str) -> None:
        text = str(body)
        if text in seen:
            return
        seen.add(text)
        out.append(
            {
                "tags": tuple(str(tag) for tag in tags),
                "body": text,
                "slot": "loop_" + str(effect),
                "needs": str(needs),
                "effect": str(effect),
            }
        )

    # (a) filtered sequence and index accumulation ---------------------
    for seq in seqs:
        add(("filter", "positive", "sequence"), (
            "return [x for x in %s if x > 0]" % seq), "filter", "list")
        add(("filter", "negative", "sequence"), (
            "return [x for x in %s if x < 0]" % seq), "filter", "list")
        add(("filter", "even", "sequence"), (
            "return [x for x in %s if x %% 2 == 0]" % seq), "filter", "list")
        add(("filter", "odd", "sequence"), (
            "return [x for x in %s if x %% 2 != 0]" % seq), "filter", "list")
        add(("unique", "distinct", "sequence"), (
            "return list(dict.fromkeys(%s))" % seq), "filter", "list")
        add(("count", "length", "distinct", "sequence"), (
            "return len(set(%s))" % seq), "fold", "int")
        add(("count", "occurrence", "frequency", "sequence"), (
            "counts = {}\n"
            "for x in %s:\n"
            "    counts[x] = counts.get(x, 0) + 1\n"
            "return counts" % seq), "dict", "dict")
        add(("count", "most", "frequency", "mode", "sequence"), (
            "counts = {}\n"
            "for x in %s:\n"
            "    counts[x] = counts.get(x, 0) + 1\n"
            "return max(counts, key=counts.get)" % seq), "fold", "any")
        add(("count", "sum", "accumulate", "fold", "sequence"), (
            "total = 0\n"
            "for x in %s:\n"
            "    total += x\n"
            "return total" % seq), "fold", "int")
        add(("product", "multiply", "fold", "sequence"), (
            "total = 1\n"
            "for x in %s:\n"
            "    total *= x\n"
            "return total" % seq), "fold", "int")
        add(("sum", "product", "tuple", "sequence"), (
            "total = 0\n"
            "product = 1\n"
            "for x in %s:\n"
            "    total += x\n"
            "    product *= x\n"
            "return (total, product)" % seq), "tuple", "tuple")
        add(("max", "min", "range", "sequence"), (
            "return max(%s) - min(%s)" % (seq, seq)), "fold", "int")
        add(("elementwise", "map", "scale", "sequence"), (
            "low = min(%s)\n"
            "high = max(%s)\n"
            "if high == low:\n"
            "    return [0.0 for x in %s]\n"
            "return [(x - low) / (high - low) for x in %s]" % (seq, seq, seq, seq)),
            "map", "list")
        add(("elementwise", "running", "total", "prefix", "sequence"), (
            "out = []\n"
            "total = 0\n"
            "for x in %s:\n"
            "    total += x\n"
            "    out.append(total)\n"
            "return out" % seq), "map", "list")
        add(("elementwise", "running", "maximum", "rolling", "sequence"), (
            "out = []\n"
            "best = None\n"
            "for x in %s:\n"
            "    best = x if best is None or x > best else best\n"
            "    out.append(best)\n"
            "return out" % seq), "map", "list")
        add(("elementwise", "halve", "half", "sequence"), (
            "return [x / 2 for x in %s]" % seq), "map", "list")
        add(("elementwise", "square", "sequence"), (
            "return [x * x for x in %s]" % seq), "map", "list")
        add(("elementwise", "double", "sequence"), (
            "return [x + x for x in %s]" % seq), "map", "list")
        add(("sort", "order", "sequence"), "return sorted(%s)" % seq, "sort", "list")
        add(("reverse", "order", "sequence"), "return list(reversed(%s))" % seq, "sort", "list")
        add(("median", "sort", "middle", "sequence"), (
            "ordered = sorted(%s)\n"
            "middle = len(ordered) // 2\n"
            "return ordered[middle]" % seq), "sort", "any")
        add(("first", "head", "sequence"),
             "return %s[0] if %s else None" % (seq, seq), "fold", "any")
        add(("last", "tail", "sequence"),
             "return %s[-1] if %s else None" % (seq, seq), "fold", "any")
        add(("filter", "index", "position", "sequence"), (
            "out = []\n"
            "for index, x in enumerate(%s):\n"
            "    if x:\n"
            "        out.append(index)\n"
            "return out" % seq), "filter", "list")

    # (b) adjacent-pair, two-pointer and window scans --------------------
    for seq in seqs:
        for threshold_name in nums:
            # Cycle AB2: "any two elements closer than the threshold" is the
            # canonical close-pair predicate.  The AA2 sketch compared only
            # ADJACENT elements, which passes a tiny public example and fails
            # on a hidden list whose close pair is not adjacent.  The nested
            # pair scan is the total predicate.
            add(("pairwise", "close", "threshold", "all", "any", "sequence"), (
                "for i in range(len(%s)):\n"
                "    for j in range(i + 1, len(%s)):\n"
                "        if abs(%s[i] - %s[j]) < %s:\n"
                "            return True\n"
                "return False" % (seq, seq, seq, seq, threshold_name)), "pairwise", "bool")
            add(("pairwise", "adjacent", "close", "threshold", "sequence"), (
                "for i in range(len(%s) - 1):\n"
                "    if abs(%s[i] - %s[i + 1]) < %s:\n"
                "        return True\n"
                "return False" % (seq, seq, seq, threshold_name)), "pairwise", "bool")
        add(("pairwise", "adjacent", "delta", "diff", "sequence"), (
            "return [abs(%s[i] - %s[i + 1]) for i in range(len(%s) - 1)]"
            % (seq, seq, seq)), "pairwise", "list")
        add(("pairwise", "adjacent", "delta", "diff", "sequence"), (
            "return [%s[i + 1] - %s[i] for i in range(len(%s) - 1)]"
            % (seq, seq, seq)), "pairwise", "list")
        add(("pairwise", "adjacent", "tuple", "sequence"), (
            "return [(%s[i], %s[i + 1]) for i in range(len(%s) - 1)]"
            % (seq, seq, seq)), "pairwise", "list")
        add(("pairwise", "sort", "closest", "nearest", "sequence"), (
            "ordered = sorted(%s)\n"
            "best = None\n"
            "for i in range(len(ordered) - 1):\n"
            "    gap = ordered[i + 1] - ordered[i]\n"
            "    if best is None or gap < best[0]:\n"
            "        best = (gap, ordered[i], ordered[i + 1])\n"
            "if best is None:\n"
            "    return ()\n"
            "return (best[1], best[2])" % seq), "pairwise", "tuple")
        add(("two_pointer", "pair", "sum", "zero", "sequence"), (
            "for i in range(len(%s)):\n"
            "    for j in range(i + 1, len(%s)):\n"
            "        if %s[i] + %s[j] == 0:\n"
            "            return True\n"
            "return False" % (seq, seq, seq, seq)), "pairwise", "bool")
        add(("two_pointer", "pair", "sum", "tuple", "sequence"), (
            "for i in range(len(%s)):\n"
            "    for j in range(i + 1, len(%s)):\n"
            "        if %s[i] + %s[j] == 0:\n"
            "            return (%s[i], %s[j])\n"
            "return None" % (seq, seq, seq, seq, seq, seq)), "pairwise", "tuple")
        add(("triple", "sum", "zero", "sequence"), (
            "for i in range(len(%s)):\n"
            "    for j in range(i + 1, len(%s)):\n"
            "        for k in range(j + 1, len(%s)):\n"
            "            if %s[i] + %s[j] + %s[k] == 0:\n"
            "                return True\n"
            "return False" % (seq, seq, seq, seq, seq, seq)), "pairwise", "bool")
        add(("window", "maximum", "sliding", "pairwise", "sequence"), (
            "out = []\n"
            "for i in range(len(%s) - 1):\n"
            "    if %s[i] > %s[i + 1]:\n"
            "        return False\n"
            "return True" % (seq, seq, seq)), "pairwise", "bool")
        add(("consecutive", "duplicate", "adjacent", "sequence"), (
            "out = []\n"
            "for x in %s:\n"
            "    if not out or out[-1] != x:\n"
            "        out.append(x)\n"
            "return out" % seq), "filter", "list")
        add(("duplicate", "repeated", "frequency", "sequence"), (
            "counts = {}\n"
            "for x in %s:\n"
            "    counts[x] = counts.get(x, 0) + 1\n"
            "return [x for x in counts if counts[x] > 1]" % seq), "filter", "list")

    # (c) two-sequence and substring predicates --------------------------
    for left in seqs:
        for right in seqs:
            if left == right:
                continue
            add(("intersection", "common", "membership", "sequence"), (
                "return [x for x in %s if x in %s]" % (left, right)), "filter", "list")
            add(("difference", "absent", "membership", "sequence"), (
                "return [x for x in %s if x not in %s]" % (left, right)), "filter", "list")
            add(("union", "merge", "membership", "sequence"), (
                "return list(dict.fromkeys(list(%s) + list(%s)))" % (left, right)),
                "filter", "list")
    for seq in seqs:
        for text in strs:
            add(("filter", "substring", "contains", "string"), (
                "return [x for x in %s if %s in x]" % (seq, text)), "filter", "list")

    # (d) stateful string transduction -----------------------------------
    for text in strs:
        add(("string", "vowel", "remove", "filter"), (
            "out = []\n"
            "for ch in %s:\n"
            "    if ch not in \"aeiouAEIOU\":\n"
            "        out.append(ch)\n"
            "return \"\".join(out)" % text), "filter", "str")
        add(("string", "space", "remove", "filter"), (
            "out = []\n"
            "for ch in %s:\n"
            "    if not ch.isspace():\n"
            "        out.append(ch)\n"
            "return \"\".join(out)" % text), "filter", "str")
        add(("string", "palindrome", "reverse"), (
            "return %s == %s[::-1]" % (text, text)), "predicate", "bool")
        add(("string", "reverse", "order"), (
            "return %s[::-1]" % text), "sort", "str")
        add(("string", "lower", "case", "distinct"), (
            "return len(set(%s.lower()))" % text), "fold", "int")
        add(("string", "upper", "case"), "return %s.upper()" % text, "map", "str")
        add(("string", "lower", "case"), "return %s.lower()" % text, "map", "str")
        add(("string", "count", "character", "vowel"), (
            "count = 0\n"
            "for ch in %s:\n"
            "    if ch in \"aeiouAEIOU\":\n"
            "        count += 1\n"
            "return count" % text), "fold", "int")
        add(("string", "buffer", "alternate", "case"), (
            "out = []\n"
            "for index, ch in enumerate(%s):\n"
            "    out.append(ch.lower() if index %% 2 == 0 else ch.upper())\n"
            "return \"\".join(out)" % text), "map", "str")
        add(("string", "strip", "space", "split"), (
            "return \" \".join(%s.split())" % text), "filter", "str")
        add(("string", "prefix", "startswith"), (
            "return %s.startswith(%s[:1])" % (text, text)), "predicate", "bool")
        for other in strs:
            if other == text:
                continue
            add(("string", "count", "substring", "occurrence"), (
                "count = 0\n"
                "for i in range(len(%s) - len(%s) + 1):\n"
                "    if %s[i:i + len(%s)] == %s:\n"
                "        count += 1\n"
                "return count" % (text, other, text, other, other)), "fold", "int")
            add(("string", "find", "substring", "index"), (
                "return %s.find(%s)" % (text, other)), "fold", "int")

    # (e) bounded arithmetic, divisors and modular recurrence ------------
    for num in nums:
        add(("prime", "predicate", "number"), (
            "if %s < 2:\n"
            "    return False\n"
            "for i in range(2, int(%s ** 0.5) + 1):\n"
            "    if %s %% i == 0:\n"
            "        return False\n"
            "return True" % (num, num, num)), "predicate", "bool")
        add(("divisor", "largest", "number"), (
            "for i in range(%s - 1, 0, -1):\n"
            "    if %s %% i == 0:\n"
            "        return i\n"
            "return 1" % (num, num)), "fold", "int")
        add(("factor", "factorize", "prime", "number"), (
            "out = []\n"
            "n = %s\n"
            "i = 2\n"
            "while i * i <= n:\n"
            "    while n %% i == 0:\n"
            "        out.append(i)\n"
            "        n //= i\n"
            "    i += 1\n"
            "if n > 1:\n"
            "    out.append(n)\n"
            "return out" % num), "fold", "list")
        add(("count", "prime", "below", "number"), (
            "count = 0\n"
            "for value in range(2, %s):\n"
            "    if all(value %% d for d in range(2, value)):\n"
            "        count += 1\n"
            "return count" % num), "fold", "int")
        add(("fibonacci", "recurrence", "number"), (
            "a, b = 0, 1\n"
            "for _ in range(%s):\n"
            "    a, b = b, a + b\n"
            "return a" % num), "recurrence", "int")
        add(("modular", "power", "number"), (
            "return pow(2, %s, %s)" % (num, num)), "recurrence", "int")
        add(("square", "number"), "return %s * %s" % (num, num), "fold", "int")
        add(("increment", "number"), "return %s + 1" % num, "fold", "int")
        add(("even", "predicate", "number"), "return %s %% 2 == 0" % num, "predicate", "bool")
        add(("factorial", "recurrence", "number"), (
            "total = 1\n"
            "for i in range(2, %s + 1):\n"
            "    total *= i\n"
            "return total" % num), "recurrence", "int")
        for other in nums:
            if other == num:
                continue
            add(("gcd", "divisor", "number"), (
                "a, b = abs(%s), abs(%s)\n"
                "while b:\n"
                "    a, b = b, a %% b\n"
                "return a" % (num, other)), "recurrence", "int")
            add(("sum", "add", "number"), "return %s + %s" % (num, other), "fold", "int")
            add(("base", "convert", "number"), (
                "digits = \"0123456789abcdefghijklmnopqrstuvwxyz\"\n"
                "value = %s\n"
                "base = %s\n"
                "if value == 0:\n"
                "    return \"0\"\n"
                "out = \"\"\n"
                "while value > 0:\n"
                "    out = digits[value %% base] + out\n"
                "    value //= base\n"
                "return out" % (num, other)), "convert", "str")

    # (a2) accumulator predicates and small reductions ----------------
    for seq in seqs:
        add(("below", "threshold", "predicate", "all", "sequence"), (
            "for x in %s:\n"
            "    if x >= %s:\n"
            "        return False\n"
            "return True" % (seq, nums[0] if nums else "0")), "predicate", "bool")
        add(("negative", "running", "accumulate", "prefix", "sequence"), (
            "total = 0\n"
            "for x in %s:\n"
            "    total += x\n"
            "    if total < 0:\n"
            "        return True\n"
            "return False" % seq), "predicate", "bool")
        add(("sum", "absolute", "magnitude", "sequence"), (
            "total = 0\n"
            "for x in %s:\n"
            "    total += abs(x)\n"
            "return total" % seq), "fold", "int")
        add(("sum", "square", "sequence"), (
            "total = 0\n"
            "for x in %s:\n"
            "    total += x * x\n"
            "return total" % seq), "fold", "int")
        add(("index", "first", "position", "find", "sequence"), (
            "for index, x in enumerate(%s):\n"
            "    if x == %s:\n"
            "        return index\n"
            "return -1" % (seq, nums[0] if nums else "None")), "fold", "int")
        add(("longest", "run", "consecutive", "sequence"), (
            "best = 0\n"
            "current = 0\n"
            "previous = None\n"
            "for x in %s:\n"
            "    current = current + 1 if x == previous else 1\n"
            "    previous = x\n"
            "    best = current if current > best else best\n"
            "return best" % seq), "fold", "int")
        add(("max", "position", "index", "sequence"), (
            "best = 0\n"
            "for index, x in enumerate(%s):\n"
            "    if x > %s[best]:\n"
            "        best = index\n"
            "return best" % (seq, seq)), "fold", "int")
        add(("sorted", "length", "order", "sequence"), (
            "return sorted(%s, key=len)" % seq), "sort", "list")
        add(("count", "even", "odd", "parity", "sequence"), (
            "count = 0\n"
            "for x in %s:\n"
            "    if x %% 2 == 0:\n"
            "        count += 1\n"
            "return count" % seq), "fold", "int")
        add(("chunk", "pair", "group", "sequence"), (
            "return [%s[i:i + 2] for i in range(0, len(%s), 2)]" % (seq, seq)),
            "stride", "list")
        add(("balance", "bracket", "parenthesis", "stack", "string"), (
            "depth = 0\n"
            "groups = []\n"
            "current = []\n"
            "for ch in %s:\n"
            "    if ch == \"(\":\n"
            "        depth += 1\n"
            "    if depth > 0:\n"
            "        current.append(ch)\n"
            "    if ch == \")\":\n"
            "        depth -= 1\n"
            "        if depth == 0:\n"
            "            groups.append(\"\".join(current))\n"
            "            current = []\n"
            "return groups" % (strs[0] if strs else seq)), "filter", "list")

    # sort every third stride (a stride scan over one sequence)
    for seq in seqs:
        # Cycle AB2: "sort every third element" sorts the indices 0, 3, 6, ...
        # and leaves the other two residue classes in place.  The AA2 sketch
        # sorted all three residue classes, which agrees on a public example
        # whose other residues are already ordered and disagrees on the
        # hidden list.
        add(("sort", "stride", "third", "every", "sequence"), (
            "out = list(%s)\n"
            "block = sorted(out[0::3])\n"
            "out[0::3] = block\n"
            "return out" % seq), "stride", "list")
        add(("sort", "stride", "third", "sequence"), (
            "out = list(%s)\n"
            "for start in range(3):\n"
            "    block = sorted(out[start::3])\n"
            "    out[start::3] = block\n"
            "return out" % seq), "stride", "list")

    # (f) nested pair / window scans (Cycle AB2) -------------------------
    for seq in seqs:
        for width in nums:
            add(("window", "sliding", "maximum", "nested", "sequence"), (
                "width = int(%s)\n"
                "out = []\n"
                "for i in range(len(%s) - width + 1):\n"
                "    out.append(max(%s[i:i + width]))\n"
                "return out" % (width, seq, seq)), "pairwise", "list")
            add(("window", "sliding", "count", "nested", "sequence"), (
                "width = int(%s)\n"
                "out = 0\n"
                "for i in range(len(%s) - width + 1):\n"
                "    if sum(%s[i:i + width]) > 0:\n"
                "        out += 1\n"
                "return out" % (width, seq, seq)), "pairwise", "int")
        add(("pair", "nested", "closest", "distance", "sequence"), (
            "best = None\n"
            "for i in range(len(%s)):\n"
            "    for j in range(i + 1, len(%s)):\n"
            "        gap = abs(%s[i] - %s[j])\n"
            "        if best is None or gap < best:\n"
            "            best = gap\n"
            "return best" % (seq, seq, seq, seq)), "pairwise", "any")

    # (g) prefix-sum / maximum subarray / monotonic stack (Cycle AB2) ----
    for seq in seqs:
        add(("subarray", "maximum", "contiguous", "sum", "kadane", "sequence"), (
            "best = %s[0]\n"
            "current = %s[0]\n"
            "for x in %s[1:]:\n"
            "    current = x if current < 0 else current + x\n"
            "    if current > best:\n"
            "        best = current\n"
            "return best" % (seq, seq, seq)), "fold", "int")
        add(("subarray", "maximum", "sum", "nonempty", "sequence"), (
            "best = current = 0\n"
            "for x in %s:\n"
            "    current = max(0, current + x)\n"
            "    best = max(best, current)\n"
            "return best" % seq), "fold", "int")
        add(("next", "greater", "monotonic", "stack", "sequence"), (
            "stack = []\n"
            "out = [-1 for _ in %s]\n"
            "for index, x in enumerate(%s):\n"
            "    while stack and %s[stack[-1]] < x:\n"
            "        out[stack.pop()] = index\n"
            "    stack.append(index)\n"
            "return out" % (seq, seq, seq)), "pairwise", "list")
        add(("prefix", "sum", "cumulative", "sequence"), (
            "out = []\n"
            "total = 0\n"
            "for x in %s:\n"
            "    total += x\n"
            "    out.append(total)\n"
            "return out" % seq), "map", "list")

    # (h) sorting-key projection and memoised recurrence (Cycle AB2) -----
    for seq in seqs:
        add(("sort", "key", "projection", "order", "sequence"), (
            "return sorted(%s, key=str)" % seq),
            "sort", "list")
        add(("sort", "key", "absolute", "magnitude", "sequence"), (
            "return sorted(%s, key=abs)" % seq), "sort", "list")
    for text in strs:
        add(("sort", "key", "length", "string", "sequence"), (
            "return sorted(%s, key=len)" % text), "sort", "list")
        add(("sort", "key", "lower", "case", "string"), (
            "return sorted(%s, key=lambda x: (str(x).lower(), str(x)))" % text),
            "sort", "list")
    for number in nums:
        add(("recurrence", "memoised", "fibonacci", "loop", "int"), (
            "memo = {0: 0, 1: 1}\n"
            "for index in range(2, int(%s) + 1):\n"
            "    memo[index] = memo[index - 1] + memo[index - 2]\n"
            "return memo[int(%s)]" % (number, number)), "fold", "int")
    return out
