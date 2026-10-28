#!/usr/bin/env python3
"""TimescaleDB ledger gate: write, restart the container, verify survival.

Two phases so the shell can restart the container in between.

    python scripts/verify_timescale_ledger.py write
    docker restart henri-timescaledb
    python scripts/verify_timescale_ledger.py verify

Counted gates:
  TS-1  rows persist across a real container restart (count matches the receipt)
  TS-2  uint8 bins survive byte-exact (sha256 of the concatenated blob)
  TS-3  the hypertable exists and has a time dimension
  TS-4  rehydrated Stiefel charts keep c_max < 1e-6
  TS-5  storage recall returns the stored label for every record
  TS-6  kill control: an unrelated query must NOT recall a stored label
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from henri.holographic_ledger import LedgerRecord
from henri.timescale_ledger import TimescaleLedgerStore, rehydrate

RECEIPT = r"C:/Users/chan/AppData/Local/hermes/cache/scratch/timescale_receipt.json"
D_MODEL = 4096
N = 64
RESULTS = []


def record(gate: str, ok: bool, detail: str) -> None:
    RESULTS.append((gate, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate}: {detail}")


def unit(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


def phase_write() -> int:
    from henri.holographic_ledger import HolographicAgenticLedger

    rng = np.random.default_rng(4242)
    store = TimescaleLedgerStore(d_model=D_MODEL, recreate=True)
    store.reset()
    led = HolographicAgenticLedger(d_model=D_MODEL, store=store)

    digests = []
    for m in range(N):
        k = unit(D_MODEL, rng)
        rec = led.commit(k, k, label=f"engram_{m:03d}")
        digests.append(hashlib.sha256(rec.bins.tobytes()).hexdigest())

    blob = hashlib.sha256("".join(digests).encode()).hexdigest()
    receipt = {
        "n": store.count(),
        "d_model": D_MODEL,
        "blob_sha256": blob,
        "labels": [f"engram_{m:03d}" for m in range(N)],
    }
    with open(RECEIPT, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"WROTE n={receipt['n']} blob_sha256={blob[:16]}... -> {RECEIPT}")
    store.close()
    return 0


def phase_verify() -> int:
    with open(RECEIPT, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    store = TimescaleLedgerStore(d_model=D_MODEL)
    rows = store.fetch_all()
    labels_db = [r.label for r in rows]

    record(
        "TS-1 restart survival",
        len(rows) == receipt["n"],
        f"rows={len(rows)} expected={receipt['n']} after `docker restart henri-timescaledb`",
    )

    digests = [hashlib.sha256(r.bins.tobytes()).hexdigest() for r in rows]
    blob = hashlib.sha256("".join(digests).encode()).hexdigest()
    record(
        "TS-2 byte-exact bins",
        blob == receipt["blob_sha256"] and rows[0].bins.dtype == np.uint8,
        f"blob_sha256 match={blob == receipt['blob_sha256']} dtype={rows[0].bins.dtype}",
    )

    with store._conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM timescaledb_information.hypertables "
            "WHERE hypertable_name = 'henri_ledger'"
        )
        ht = int(cur.fetchone()[0])
        cur.execute(
            "SELECT count(*) FROM timescaledb_information.dimensions "
            "WHERE hypertable_name = 'henri_ledger' AND dimension_type = 'Time'"
        )
        dims = int(cur.fetchone()[0])
    record("TS-3 hypertable", ht == 1 and dims == 1, f"hypertables={ht} time_dims={dims}")

    led = rehydrate(rows, d_model=D_MODEL)
    cmax = led.max_coherence()
    record("TS-4 rehydrated Stiefel", cmax < 1e-6, f"c_max={cmax:.2e} (bar 1e-6), engrams={led.size}")

    hits = 0
    for rec in rows:
        wave = led.dequantize(rec.bins)
        got = led.query(wave)
        if led.memory.charts[got.chart_id].labels[got.local_index] == rec.label:
            hits += 1
    record(
        "TS-5 storage recall",
        hits == len(rows),
        f"recalled={hits}/{len(rows)} after deserialisation from BYTEA",
    )

    rng = np.random.default_rng(99)
    miss = led.query(unit(D_MODEL, rng))
    record(
        "TS-6 kill control",
        miss.margin < 0.10,
        f"unrelated-query margin={miss.margin:.4f} (bar <0.10); a metric that cannot fail is decoration",
    )

    store.close()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n=== {passed}/{len(RESULTS)} TimescaleDB gates passed ===")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "write"
    sys.exit(phase_write() if mode == "write" else phase_verify())
