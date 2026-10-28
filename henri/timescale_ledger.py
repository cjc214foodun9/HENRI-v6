"""TimescaleDB-backed ledger store.

Real hypertable, real restart survival, typed refusal when the service is
absent. Credentials come from the DSN; the default DSN is local-only and the
compose file pins the same value. No secret is stored in this module.

The store satisfies the `LedgerStore` protocol in `henri.holographic_ledger`:
`append`, `fetch_all`, `count`. `rehydrate` rebuilds an in-memory HHRG ledger
from persisted rows, so a container restart cannot silently lose state.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

from henri.holographic_ledger import LedgerRecord

DEFAULT_DSN = os.environ.get(
    "HENRI_TSDB_DSN",
    "postgresql://postgres:henri_local_only@127.0.0.1:55432/henri",
)


class TimescaleLedgerStore:
    """Durable ledger rows in a TimescaleDB hypertable."""

    def __init__(
        self,
        dsn: str = DEFAULT_DSN,
        d_model: int = 4096,
        migrate: bool = True,
        recreate: bool = False,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "BLOCKED: psycopg is not installed. Install `psycopg[binary]`. "
                "Do not substitute a file-backed store for a database."
            ) from exc
        self.dsn = dsn
        self.d_model = int(d_model)
        try:
            self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
        except Exception as exc:
            raise RuntimeError(f"BLOCKED: TimescaleDB is unreachable at {dsn!r}: {exc}") from exc
        if migrate:
            self._migrate(recreate=recreate)

    def _migrate(self, recreate: bool = False) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            if recreate:
                cur.execute("DROP TABLE IF EXISTS henri_ledger CASCADE")
            # TimescaleDB requires the partitioning column inside every unique
            # index. `PRIMARY KEY (record_id)` alone makes create_hypertable fail
            # with "cannot create a unique index without the column ts".
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS henri_ledger (
                    record_id   BIGINT NOT NULL,
                    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
                    bins        BYTEA NOT NULL,
                    label       TEXT NOT NULL,
                    chart_id    INTEGER NOT NULL,
                    local_index INTEGER NOT NULL,
                    PRIMARY KEY (record_id, ts)
                )
                """
            )
            cur.execute(
                "SELECT create_hypertable('henri_ledger','ts',"
                " if_not_exists => TRUE, migrate_data => TRUE)"
            )

    # ------------------------------------------------------------- protocol
    def append(self, record: LedgerRecord) -> None:
        bins = np.asarray(record.bins)
        if bins.dtype != np.uint8:
            raise ValueError("ledger bins must be uint8")
        if bins.size != self.d_model:
            raise ValueError(f"expected {self.d_model} channels, got {bins.size}")
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO henri_ledger (record_id, bins, label, chart_id, local_index)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (record.record_id, bins.tobytes(), record.label, record.chart_id, record.local_index),
            )

    def fetch_all(self) -> List[LedgerRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT record_id, bins, label, chart_id, local_index "
                "FROM henri_ledger ORDER BY record_id"
            )
            rows = cur.fetchall()
        out: List[LedgerRecord] = []
        for rid, blob, label, chart_id, local_index in rows:
            out.append(
                LedgerRecord(
                    record_id=int(rid),
                    bins=np.frombuffer(bytes(blob), dtype=np.uint8).copy(),
                    label=str(label),
                    chart_id=int(chart_id),
                    local_index=int(local_index),
                )
            )
        return out

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM henri_ledger")
            return int(cur.fetchone()[0])

    # ------------------------------------------------------------- test hooks
    def reset(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("TRUNCATE henri_ledger")

    def close(self) -> None:
        self._conn.close()


def rehydrate(records: Sequence[LedgerRecord], d_model: int, chart_capacity: int = 256):
    """Rebuild an in-memory HHRG ledger from persisted rows."""
    from henri.holographic_ledger import HolographicAgenticLedger, InMemoryLedgerStore

    led = HolographicAgenticLedger(
        d_model=d_model, chart_capacity=chart_capacity, store=InMemoryLedgerStore()
    )
    for rec in records:
        wave = led.dequantize(rec.bins)
        led.commit(wave, wave, label=rec.label)
    return led
