"""Compute the two slot digests under the pinned encoder, and report the
environment that produced them.  Run from the v6 checkout.

    python scripts/record_embedding_pins.py --out tests/pins/embedding_pins.json

Without ``--out`` it prints the payload and changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from henri import operator_ingress as oi
from henri.semantic_backbone import MODEL_NAME, MODEL_REVISION, SemanticBackbone

REV_LIST = "Write a function to reverse a list"
SORT_LIST = "Write a function to sort a list"
CODE_REV = "def f(s):\n    return s[::-1]\n"
CODE_SORT = "def f(s):\n    return ''.join(sorted(s))\n"
SUM_DOC = "Write a function to find the sum of a and b"
DIFF_DOC = "Write a function to find the difference of a and b"
ADD_CODE = "def f(a, b):\n    return a + b\n"
SUB_CODE = "def f(a, b):\n    return a - b\n"

PIN_FIXTURES = (
    SUM_DOC, DIFF_DOC, REV_LIST, SORT_LIST,
    "return a function of two values",
    ADD_CODE, SUB_CODE, CODE_REV, CODE_SORT,
    "the cat sat on the mat",
)

bb = SemanticBackbone()
aware = oi.OperatorAwareBackbone(base=bb)
stem = getattr(bb, "stem", None)
kind = getattr(stem, "kind", "unknown")
if kind != "sentence_transformer":
    print(json.dumps({"error": f"encoder is {kind}, not sentence_transformer; refusing to pin"}))
    sys.exit(2)


def digest(lo: int, hi: int) -> str:
    h = hashlib.sha256()
    for text in PIN_FIXTURES:
        h.update(np.ascontiguousarray(aware.encode(text).psi[lo:hi]).tobytes())
    return h.hexdigest()


import sentence_transformers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402


def _head_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


payload = {
    "schema": 1,
    "notes": (
        "Byte-identity pins for the two slot regions. A pin is only meaningful inside "
        "the environment that produced it, so the consuming test skips with a named "
        "reason when the encoder revision or a library version differs. Regenerate "
        "deliberately with scripts/record_embedding_pins.py."
    ),
    "model": MODEL_NAME,
    "revision": MODEL_REVISION,
    "encoder_kind": kind,
    "dim_sem": int(getattr(stem, "dim_sem", 0)),
    "versions": {
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    },
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "recorded_from_git": _head_sha(),
    "digests": {
        "slot1_16384_32768": digest(16384, 32768),
        "slot2_32768_49152": digest(32768, 49152),
    },
}


if "--out" in sys.argv:
    target = Path(sys.argv[sys.argv.index("--out") + 1])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
else:
    print(json.dumps(payload, indent=2))
