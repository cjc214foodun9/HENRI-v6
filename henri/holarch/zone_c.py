"""Zone C: holographic agentic memory on NVMe and TimescaleDB.

Evidence label: OBSERVED for the round-trip, recall, and latency numbers
the tests and receipts record; DERIVED for the storage bounds.

What this module is
-------------------
An append-only episodic store of complex wave engrams. Each row holds one
engram in the 4-slot layout of the Zone C specification:

    recorded_at, engram_id, label, dim,
    slot_0 .. slot_3   (complex64, d_slot values each),
    gram               (complex64, gram_dims projection),
    lyapunov, info_gain, tier, weight

The store is unbounded. The resident index is bounded: one complex
projection of 64 values per engram, plus an LRU wave cache of fixed
capacity. Memory therefore grows as O(N * gram_dims) bytes, not
O(N * dim), and the context tensor handed to Zone A has a FIXED shape at
every N. That is the measurable meaning of "near-infinite context" used
here: an unbounded store behind a bounded working set. Retrieval SNR is
measured, never asserted to be logarithmic.

Two backends
------------
* TimescaleEngramStore: a real hypertable, used when the DSN is reachable.
* SQLiteEngramStore: a real WAL-mode database on the local NVMe drive,
  used when TimescaleDB is absent. The choice is recorded in
  ZoneCMemory.backend and is never silent.

Both are real databases. A dictionary pretending to be a database is not
offered as a substitute.

Holography
----------
Writes bind a key phasor to the value wave (pointwise unit-modulus
binding, henri.qfhrr.pbind) and superpose the result into a running
boundary wave. A query can be answered three ways, in increasing cost:
the boundary (coarse, O(D)), the Gram index (candidate set), and exact
coherence rescoring on the full wave. The tests measure all three.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from henri.holarch.centered_charts import CenteredChartIndex
from henri.qfhrr import coherence, pbind

DEFAULT_DSN = os.environ.get(
    "HENRI_TSDB_DSN",
    "postgresql://postgres:henri_local_only@127.0.0.1:55432/henri",
)
DEFAULT_PATH = Path(
    os.environ.get("HENRI_ZONEC_PATH", "data/holarch/zone_c.sqlite3")
)
DEFAULT_TABLE = "zone_c_engram_timeseries"

_EPS = 1e-12
TIER_AXIOMATIC = 3

# Decision D1: disjoint task label blocks. The four task label blocks are the
# same blocks the Zone A swarm uses, so a stored label identifies its task
# without a schema change. Engrams that are not task engrams (symbolic atlas,
# playbooks) are stored above SYMBOL_LABEL_OFFSET.
TASK_LABEL_BLOCKS: Dict[str, Tuple[int, int]] = {
    "vision": (0, 10),
    "language": (10, 106),
    "action": (106, 110),
    "relational": (110, 118),
}
SYMBOL_LABEL_OFFSET = 10000
PLAYBOOK_LABEL_OFFSET = 20000


def task_label_block(task: str) -> Tuple[int, int]:
    """Half-open global label block for one task."""
    if task not in TASK_LABEL_BLOCKS:
        raise ValueError(f"unknown task {task!r}")
    return TASK_LABEL_BLOCKS[task]


def global_label(task: str, local: int) -> int:
    """Map a local class label into the task's disjoint global block."""
    start, end = task_label_block(task)
    value = int(local)
    if not 0 <= value < (end - start):
        raise ValueError(f"label {local} does not fit task {task!r}")
    return start + value


def local_label(task: str, label: int) -> int:
    """Inverse of global_label."""
    start, _ = task_label_block(task)
    return int(label) - start


def _unit(wave: np.ndarray) -> np.ndarray:
    arr = np.asarray(wave, dtype=np.complex64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < _EPS:
        raise ValueError("cannot store or query a zero or non-finite wave")
    return arr / norm


BOUNDARY_AXIOM_KIND = "boundary_axiom"
"""Metadata kind that turns a tier-3 engram into an active boundary axiom."""

BOUNDARY_POLARITIES = ("LAW", "ATTRACTOR", "REPULSOR")
"""LAW is a hard Sagnac reference, ATTRACTOR lowers free energy, REPULSOR raises it."""

BOUNDARY_SOURCES = (
    "human_heuristic",
    "physics_law",
    "ast_grammar",
    "test_time_learned",
    "library_prototype",
)
"""Where an axiom came from. A learned trap is never labelled as a human rule."""

AXIOM_LABEL_BASE = SYMBOL_LABEL_OFFSET + 1000
"""Boundary axioms live outside every task label block, so a task query never returns one."""


@dataclass(frozen=True)
class BoundaryAxiom:
    """One typed Zone C boundary axiom (Directive K2).

    Attributes:
        axiom_id: stable metadata key, unique per namespace.
        name: human-readable name.
        polarity: LAW, ATTRACTOR, or REPULSOR.
        source: one of BOUNDARY_SOURCES.
        slot_mask: which of the four slots the axiom governs.
        namespace: tenant namespace.
        engram_id: the tier-3 Zone C row that carries the wave.
        wave: the governed unit wave, masked to slot_mask.
        created_ts: unix seconds of registration.
    """

    axiom_id: str
    name: str
    polarity: str
    source: str
    slot_mask: Tuple[bool, bool, bool, bool]
    namespace: str
    engram_id: int
    wave: np.ndarray
    created_ts: float = 0.0

    def governed(self) -> np.ndarray:
        """The wave with every ungoverned slot zeroed and the remainder unit."""
        slots = np.asarray(self.wave, dtype=np.complex64).reshape(4, -1).copy()
        mask = np.asarray(self.slot_mask, dtype=bool)
        slots[~mask] = 0.0
        return _unit(slots.reshape(-1))

    def as_payload(self) -> Dict[str, object]:
        return {
            "axiom_id": str(self.axiom_id),
            "name": str(self.name),
            "polarity": str(self.polarity),
            "source": str(self.source),
            "slot_mask": [bool(value) for value in self.slot_mask],
            "namespace": str(self.namespace),
            "engram_id": int(self.engram_id),
            "created_ts": float(self.created_ts),
            "wave_sha256": hashlib.sha256(self.governed().tobytes()).hexdigest(),
        }


@dataclass(frozen=True)
class Engram:
    """One episodic memory row.

    Attributes:
        engram_id: monotonic identifier.
        ts: unix seconds at write time.
        label: integer task label. -1 means unlabelled.
        wave: complex64 value wave of length dim.
        gram: complex64 projection of length gram_dims.
        lyapunov: energy at convergence.
        info_gain: epistemic gain in nats.
        tier: 1 working, 2 episodic, 3 axiomatic.
        weight: viscoelastic retention weight.
    """

    engram_id: int
    ts: float
    label: int
    wave: np.ndarray
    gram: np.ndarray
    lyapunov: float
    info_gain: float
    tier: int
    weight: float
    namespace: str = "default"


class EngramStore(Protocol):
    """Durable engram storage. Every backend implements exactly this."""

    def append(self, engram: Engram) -> None: ...
    def append_many(self, engrams: Sequence[Engram]) -> None: ...
    def fetch(self, ids: Sequence[int]) -> List[Engram]: ...
    def metadata(self, dim: int) -> List[Tuple[int, float, int, bytes, int, float]]: ...
    def replace_wave(self, engram_id: int, wave: np.ndarray, gram: np.ndarray) -> None: ...
    def delete(self, ids: Sequence[int]) -> int: ...
    def count(self, dim: int) -> int: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


def _encode_wave(wave: np.ndarray) -> bytes:
    return np.asarray(wave, dtype=np.complex64).tobytes()


def _decode_wave(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.complex64).copy()


def _encode_gram(gram: np.ndarray) -> bytes:
    return np.asarray(gram, dtype=np.complex64).tobytes()


def _decode_gram(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.complex64).copy()


class SQLiteEngramStore:
    """WAL-mode SQLite engram store on the local NVMe filesystem."""

    backend = "sqlite"

    def __init__(self, path: str | os.PathLike = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS engram (
                engram_id INTEGER PRIMARY KEY,
                recorded_at REAL NOT NULL,
                label INTEGER NOT NULL,
                dim INTEGER NOT NULL,
                slot_0 BLOB NOT NULL,
                slot_1 BLOB NOT NULL,
                slot_2 BLOB NOT NULL,
                slot_3 BLOB NOT NULL,
                gram BLOB NOT NULL,
                lyapunov REAL NOT NULL,
                info_gain REAL NOT NULL,
                tier INTEGER NOT NULL,
                weight REAL NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default'
            )
            """
        )
        # Directive F1: a table written by an earlier cycle has no namespace
        # column. Add it in place with a default, so every existing row
        # migrates to the default tenant and no data is lost.
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(engram)").fetchall()
        }
        if "namespace" not in columns:
            self._conn.execute(
                "ALTER TABLE engram ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_engram_dim ON engram(dim, engram_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_engram_namespace ON engram(dim, namespace, engram_id)"
        )
        # Directive G4: sidecar metadata (tenant symbol tables, the code graph
        # and the compiled plans) lives beside the engrams, so a process that
        # restarts on this file answers without re-ingesting anything.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS engram_meta (
                namespace TEXT NOT NULL,
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (namespace, kind, key)
            )
            """
        )
        self._conn.commit()

    def _row(self, engram: Engram, dim: int) -> tuple:
        wave = np.asarray(engram.wave, dtype=np.complex64).reshape(4, -1)
        gram = _encode_gram(engram.gram)
        return (
            int(engram.engram_id),
            float(engram.ts),
            int(engram.label),
            int(dim),
            wave[0].tobytes(),
            wave[1].tobytes(),
            wave[2].tobytes(),
            wave[3].tobytes(),
            gram,
            float(engram.lyapunov),
            float(engram.info_gain),
            int(engram.tier),
            float(engram.weight),
            str(engram.namespace),
        )

    # ------------------------------------------------------------------ meta
    def put_meta(self, namespace: str, kind: str, key: str, payload: dict) -> None:
        import json as _json

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO engram_meta VALUES (?,?,?,?,?)",
                (
                    str(namespace),
                    str(kind),
                    str(key),
                    _json.dumps(payload, sort_keys=True, default=str),
                    float(time.time()),
                ),
            )
            self._conn.commit()

    def get_meta(self, namespace: str, kind: str, key: Optional[str] = None) -> List[Tuple[str, dict]]:
        import json as _json

        with self._lock:
            if key is None:
                rows = self._conn.execute(
                    "SELECT key, payload_json FROM engram_meta WHERE namespace=? AND kind=? ORDER BY key",
                    (str(namespace), str(kind)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT key, payload_json FROM engram_meta WHERE namespace=? AND kind=? AND key=?",
                    (str(namespace), str(kind), str(key)),
                ).fetchall()
        out: List[Tuple[str, dict]] = []
        for row in rows:
            try:
                out.append((str(row[0]), _json.loads(row[1])))
            except Exception:
                continue
        return out

    def meta_namespaces(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT namespace FROM engram_meta ORDER BY namespace"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def delete_meta(self, namespace: str, kind: Optional[str] = None) -> int:
        with self._lock:
            if kind is None:
                cur = self._conn.execute(
                    "DELETE FROM engram_meta WHERE namespace=?", (str(namespace),)
                )
            else:
                cur = self._conn.execute(
                    "DELETE FROM engram_meta WHERE namespace=? AND kind=?",
                    (str(namespace), str(kind)),
                )
            self._conn.commit()
        return int(cur.rowcount)

    def append(self, engram: Engram) -> None:
        self.append_many([engram])

    def append_many(self, engrams: Sequence[Engram]) -> None:
        if not engrams:
            return
        dim = int(np.asarray(engrams[0].wave).size)
        rows = [self._row(e, dim) for e in engrams]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO engram VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._conn.commit()

    def fetch(self, ids: Sequence[int]) -> List[Engram]:
        ids = [int(i) for i in ids]
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT engram_id, recorded_at, label, slot_0, slot_1, slot_2, slot_3, "
                f"gram, lyapunov, info_gain, tier, weight, namespace FROM engram "
                f"WHERE engram_id IN ({marks})",
                ids,
            ).fetchall()
        out: List[Engram] = []
        for row in rows:
            slot_bytes = row[3:7]
            wave = np.concatenate([_decode_wave(b) for b in slot_bytes]).astype(np.complex64)
            out.append(
                Engram(
                    engram_id=int(row[0]),
                    ts=float(row[1]),
                    label=int(row[2]),
                    wave=wave,
                    gram=_decode_gram(row[7]),
                    lyapunov=float(row[8]),
                    info_gain=float(row[9]),
                    tier=int(row[10]),
                    weight=float(row[11]),
                    namespace=str(row[12]),
                )
            )
        by_id = {e.engram_id: e for e in out}
        return [by_id[i] for i in ids if i in by_id]

    def metadata(self, dim: int, namespace: Optional[str] = None) -> List[Tuple]:
        with self._lock:
            if namespace is None:
                rows = self._conn.execute(
                    "SELECT engram_id, recorded_at, label, gram, tier, weight, namespace "
                    "FROM engram WHERE dim = ? ORDER BY engram_id",
                    (int(dim),),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT engram_id, recorded_at, label, gram, tier, weight, namespace "
                    "FROM engram WHERE dim = ? AND namespace = ? ORDER BY engram_id",
                    (int(dim), str(namespace)),
                ).fetchall()
        return [
            (int(r[0]), float(r[1]), int(r[2]), bytes(r[3]), int(r[4]), float(r[5]), str(r[6]))
            for r in rows
        ]

    def replace_wave(self, engram_id: int, wave: np.ndarray, gram: np.ndarray) -> None:
        slots = np.asarray(wave, dtype=np.complex64).reshape(4, -1)
        with self._lock:
            self._conn.execute(
                "UPDATE engram SET slot_0=?, slot_1=?, slot_2=?, slot_3=?, gram=? "
                "WHERE engram_id=?",
                (
                    slots[0].tobytes(),
                    slots[1].tobytes(),
                    slots[2].tobytes(),
                    slots[3].tobytes(),
                    _encode_gram(gram),
                    int(engram_id),
                ),
            )
            self._conn.commit()

    def delete(self, ids: Sequence[int]) -> int:
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        marks = ",".join("?" for _ in ids)
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM engram WHERE engram_id IN ({marks})", ids
            )
            self._conn.commit()
        return int(cur.rowcount)

    def count(self, dim: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*) FROM engram WHERE dim = ?", (int(dim),)
            ).fetchone()
        return int(row[0])

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM engram")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class TimescaleEngramStore:
    """A real TimescaleDB hypertable for episodic engrams."""

    backend = "timescaledb"

    def __init__(
        self,
        dsn: str = DEFAULT_DSN,
        table: str = DEFAULT_TABLE,
        connect_timeout: int = 5,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("BLOCKED: psycopg is not installed") from exc
        self.dsn = dsn
        self.table = str(table)
        if not self.table.replace("_", "").isalnum():
            raise ValueError("table name must be alphanumeric or underscore")
        self._lock = threading.Lock()
        try:
            self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        except Exception as exc:
            raise RuntimeError(f"BLOCKED: TimescaleDB unreachable at {dsn!r}: {exc}") from exc
        self._migrate()

    def _migrate(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    engram_id BIGINT NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL,
                    label INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    slot_0 BYTEA NOT NULL,
                    slot_1 BYTEA NOT NULL,
                    slot_2 BYTEA NOT NULL,
                    slot_3 BYTEA NOT NULL,
                    gram BYTEA NOT NULL,
                    lyapunov REAL NOT NULL,
                    info_gain REAL NOT NULL,
                    tier SMALLINT NOT NULL,
                    weight REAL NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    PRIMARY KEY (engram_id, recorded_at)
                )
                """
            )
            # Directive F1: an existing hypertable keeps its rows. The column
            # is added with the default tenant, so nothing is rewritten.
            cur.execute(
                f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS "
                "namespace TEXT NOT NULL DEFAULT 'default'"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_namespace "
                f"ON {self.table}(dim, namespace, engram_id)"
            )
            # Directive G4: the same metadata table, so both backends restore
            # the same sidecar state after a restart.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS engram_meta (
                    namespace TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (namespace, kind, key)
                )
                """
            )
            cur.execute(
                f"SELECT create_hypertable('{self.table}','recorded_at',"
                " chunk_time_interval => INTERVAL '1 hour',"
                " if_not_exists => TRUE, migrate_data => TRUE)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_dim ON {self.table}(dim, engram_id)"
            )

    def _params(self, engram: Engram, dim: int) -> tuple:
        slots = np.asarray(engram.wave, dtype=np.complex64).reshape(4, -1)
        return (
            int(engram.engram_id),
            float(engram.ts),
            int(engram.label),
            int(dim),
            slots[0].tobytes(),
            slots[1].tobytes(),
            slots[2].tobytes(),
            slots[3].tobytes(),
            _encode_gram(engram.gram),
            float(engram.lyapunov),
            float(engram.info_gain),
            int(engram.tier),
            float(engram.weight),
            str(engram.namespace),
        )

    # ------------------------------------------------------------------ meta
    def put_meta(self, namespace: str, kind: str, key: str, payload: dict) -> None:
        import json as _json

        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO engram_meta VALUES (%s,%s,%s,%s,now()) "
                "ON CONFLICT (namespace, kind, key) DO UPDATE SET "
                "payload_json = EXCLUDED.payload_json, updated_at = now()",
                (str(namespace), str(kind), str(key), _json.dumps(payload, sort_keys=True, default=str)),
            )

    def get_meta(self, namespace: str, kind: str, key: Optional[str] = None) -> List[Tuple[str, dict]]:
        import json as _json

        with self._lock, self._conn.cursor() as cur:
            if key is None:
                cur.execute(
                    "SELECT key, payload_json FROM engram_meta WHERE namespace=%s AND kind=%s ORDER BY key",
                    (str(namespace), str(kind)),
                )
            else:
                cur.execute(
                    "SELECT key, payload_json FROM engram_meta "
                    "WHERE namespace=%s AND kind=%s AND key=%s",
                    (str(namespace), str(kind), str(key)),
                )
            rows = cur.fetchall()
        out: List[Tuple[str, dict]] = []
        for row in rows:
            try:
                out.append((str(row[0]), _json.loads(row[1])))
            except Exception:
                continue
        return out

    def meta_namespaces(self) -> List[str]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT namespace FROM engram_meta ORDER BY namespace")
            rows = cur.fetchall()
        return [str(row[0]) for row in rows]

    def delete_meta(self, namespace: str, kind: Optional[str] = None) -> int:
        with self._lock, self._conn.cursor() as cur:
            if kind is None:
                cur.execute("DELETE FROM engram_meta WHERE namespace=%s", (str(namespace),))
            else:
                cur.execute(
                    "DELETE FROM engram_meta WHERE namespace=%s AND kind=%s",
                    (str(namespace), str(kind)),
                )
            return int(cur.rowcount)

    def append(self, engram: Engram) -> None:
        self.append_many([engram])

    def append_many(self, engrams: Sequence[Engram]) -> None:
        if not engrams:
            return
        dim = int(np.asarray(engrams[0].wave).size)
        with self._lock, self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} VALUES "
                "(%s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [self._params(e, dim) for e in engrams],
            )

    def fetch(self, ids: Sequence[int]) -> List[Engram]:
        ids = [int(i) for i in ids]
        if not ids:
            return []
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"SELECT engram_id, recorded_at, label, slot_0, slot_1, slot_2, slot_3, "
                f"gram, lyapunov, info_gain, tier, weight, namespace FROM {self.table} "
                "WHERE engram_id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall()
        out: List[Engram] = []
        for row in rows:
            wave = np.concatenate([_decode_wave(bytes(b)) for b in row[3:7]]).astype(np.complex64)
            out.append(
                Engram(
                    engram_id=int(row[0]),
                    ts=float(row[1].timestamp()),
                    label=int(row[2]),
                    wave=wave,
                    gram=_decode_gram(bytes(row[7])),
                    lyapunov=float(row[8]),
                    info_gain=float(row[9]),
                    tier=int(row[10]),
                    weight=float(row[11]),
                    namespace=str(row[12]),
                )
            )
        by_id = {e.engram_id: e for e in out}
        return [by_id[i] for i in ids if i in by_id]

    def metadata(self, dim: int, namespace: Optional[str] = None) -> List[Tuple]:
        with self._lock, self._conn.cursor() as cur:
            if namespace is None:
                cur.execute(
                    f"SELECT engram_id, EXTRACT(EPOCH FROM recorded_at), label, gram, "
                    f"tier, weight, namespace FROM {self.table} WHERE dim = %s "
                    "ORDER BY engram_id",
                    (int(dim),),
                )
            else:
                cur.execute(
                    f"SELECT engram_id, EXTRACT(EPOCH FROM recorded_at), label, gram, "
                    f"tier, weight, namespace FROM {self.table} WHERE dim = %s "
                    "AND namespace = %s ORDER BY engram_id",
                    (int(dim), str(namespace)),
                )
            rows = cur.fetchall()
        return [
            (int(r[0]), float(r[1]), int(r[2]), bytes(r[3]), int(r[4]), float(r[5]), str(r[6]))
            for r in rows
        ]

    def replace_wave(self, engram_id: int, wave: np.ndarray, gram: np.ndarray) -> None:
        slots = np.asarray(wave, dtype=np.complex64).reshape(4, -1)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {self.table} SET slot_0=%s, slot_1=%s, slot_2=%s, slot_3=%s, gram=%s "
                "WHERE engram_id=%s",
                (
                    slots[0].tobytes(),
                    slots[1].tobytes(),
                    slots[2].tobytes(),
                    slots[3].tobytes(),
                    _encode_gram(gram),
                    int(engram_id),
                ),
            )

    def delete(self, ids: Sequence[int]) -> int:
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE engram_id = ANY(%s)", (ids,))
            return int(cur.rowcount)

    def count(self, dim: int) -> int:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table} WHERE dim = %s", (int(dim),))
            return int(cur.fetchone()[0])

    def reset(self) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@dataclass
class MemoryHit:
    engram_id: int
    label: int
    score: float
    coherence: float
    ts: float
    tier: int


@dataclass
class MemoryRead:
    hits: List[MemoryHit]
    context: np.ndarray
    context_norm: float
    latency_ms: float
    candidates: int
    used_boundary: bool = False
    s_max: float = 0.0
    """Raw top-1 Gram coherence BEFORE softmax weighting.

    This is the OOD statistic of Directive 2. The softmax-weighted
    context is always unit norm, so it cannot answer 'is this query near
    the stored manifold'. The raw Gram score can.
    """
    top_wave: Optional[np.ndarray] = None
    """Full wave of the best exact-coherence hit, or None."""
    scanned: int = 0
    """Resident rows this read scored before the exact-wave rescoring.

    With use_charts=False this is the whole store. With chart routing it is
    the union of the winning charts, which is the candidate fraction Gate T1b
    and Gate T2 bound. It is a measurement, not an estimate.
    """
    probed_charts: int = 0
    """Chart count the router probed on this read (0 on a flat scan)."""
    centered: bool = False
    """True when the T2 mean-centered, whitened atlas chose the rows."""
    atlas: bool = False
    """True when the Riemannian tangent-space atlas chose the rows."""
    routing_margin: float = 0.0
    """Top-1 minus top-2 centered centroid margin on a centered read."""

    @property
    def labels(self) -> List[int]:
        return [hit.label for hit in self.hits]


class ZoneCMemory:
    """Bounded working set over an unbounded holographic engram store."""

    def __init__(
        self,
        cfg,
        store: Optional[EngramStore] = None,
        backend: str = "auto",
        path: Optional[str] = None,
        dsn: Optional[str] = None,
        table: str = DEFAULT_TABLE,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.dim = int(cfg.dim)
        # Decision C2: the wave dimension is part of the table identity. A
        # D = 1024 run and a D = 256 run can share one TimescaleDB service,
        # and neither can retrieve the other's engrams: coherence across
        # dimensions is undefined, not merely noisy.
        table = f"{table}_d{self.dim}"
        self.gram_dims = int(cfg.gram_dims)
        self.cache_capacity = int(cfg.cache_capacity)
        self.seed = int(cfg.seed if seed is None else seed)
        self.backend_reason = ""

        if store is not None:
            self.store = store
            self.backend = getattr(store, "backend", "custom")
        elif backend not in {"auto", "sqlite", "timescale", "timescaledb"}:
            raise ValueError("backend must be auto, sqlite, or timescale")
        elif backend in {"timescale", "timescaledb"}:
            self.store = TimescaleEngramStore(dsn or DEFAULT_DSN, table=table)
            self.backend = "timescaledb"
        elif backend == "sqlite":
            self.store = SQLiteEngramStore(path or DEFAULT_PATH)
            self.backend = "sqlite"
            self.backend_reason = "explicit sqlite request"
        else:
            try:
                self.store = TimescaleEngramStore(dsn or DEFAULT_DSN, table=table)
                self.backend = "timescaledb"
            except Exception as exc:
                self.store = SQLiteEngramStore(path or DEFAULT_PATH)
                self.backend = "sqlite"
                self.backend_reason = f"TimescaleDB unavailable: {type(exc).__name__}"

        rng = np.random.default_rng(self.seed)
        projection = rng.standard_normal((self.gram_dims, self.dim)) + 1j * rng.standard_normal(
            (self.gram_dims, self.dim)
        )
        projection = projection.astype(np.complex64)
        norms = np.linalg.norm(projection, axis=1, keepdims=True)
        self.projection = projection / np.maximum(norms, _EPS)

        self.context_proj = (
            rng.standard_normal((cfg.chunk, int(cfg.context_dim))) / np.sqrt(float(cfg.chunk))
        ).astype(np.float32)

        self._lock = threading.RLock()
        # Directive F1: a namespace registry, and an int code per row so the
        # filter is one vector comparison and not a per-row string compare.
        self.namespaces: List[str] = ["default"]
        self._ns_index: Dict[str, int] = {"default": 0}
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._reads = 0
        self._read_ms_total = 0.0
        self._prefetched: set = set()
        self._prefetch_hits = 0
        self._last_query: Optional[np.ndarray] = None
        self.boundary = np.zeros(self.dim, dtype=np.complex64)
        self.boundary_key = np.zeros(self.dim, dtype=np.complex64)
        self._next_id = 1
        self._size = 0
        self._capacity = 0
        # Directive S3: the two-tier spherical chart index. Charts partition the
        # resident Gram vectors into groups of at most chart_capacity engrams
        # with a unit centroid, so a query scores K centroids instead of N
        # rows and then rescans only the winning charts.
        self.chart_capacity = int(getattr(cfg, "chart_capacity", 256) or 256)
        self.chart_affinity = float(getattr(cfg, "chart_affinity", 0.10))
        self.top_charts_default = int(getattr(cfg, "top_charts", 4) or 4)
        self.charts: List[Dict[str, object]] = []
        # Directive T2: the mean-centered, whitened routing atlas. It is None
        # until build_centered_index is called for a namespace, so every
        # pre-T2 path is byte-identical. When it is synchronized and the store
        # holds at least 1024 rows, a query that does not name use_charts uses
        # it automatically.
        self.centered_index: Optional[CenteredChartIndex] = None
        self.auto_centered_threshold = 1024
        # Sovereign T2: the Riemannian tangent-space atlas. It is None until
        # build_riemannian_atlas is called; an online write can then spawn a
        # new chart without touching an existing chart frame.
        self.riemannian_atlas = None
        self.atlas_online_charts = 0
        self._centroid_matrix: Optional[np.ndarray] = None
        self._chart_of = np.full(64, -1, dtype=np.int32)
        self.chart_queries = 0
        self.chart_scanned = 0
        self.flat_scanned = 0
        self._ensure(64)
        self._load()

    # ------------------------------------------------------------- index io
    def _ns_code(self, namespace: str) -> int:
        """Stable integer code for a namespace, assigned on first sight."""
        name = str(namespace or "default")
        code = self._ns_index.get(name)
        if code is None:
            if len(self.namespaces) >= 32000:
                raise ValueError("namespace registry is full")
            code = len(self.namespaces)
            self._ns_index[name] = code
            self.namespaces.append(name)
        return int(code)

    def _ensure(self, needed: int) -> None:
        """Grow the resident index. O(N) reallocation, amortised doubling."""
        if needed <= self._capacity:
            return
        capacity = max(int(needed), max(64, self._capacity * 2))
        grown = {
            "_grams": np.zeros((capacity, self.gram_dims), dtype=np.complex64),
            "_ids": np.zeros(capacity, dtype=np.int64),
            "_ts": np.zeros(capacity, dtype=np.float64),
            "_tier": np.zeros(capacity, dtype=np.int16),
            "_weight": np.ones(capacity, dtype=np.float32),
            "_labels": np.full(capacity, -1, dtype=np.int64),
            "_ns": np.zeros(capacity, dtype=np.int16),
            "_chart_of": np.full(capacity, -1, dtype=np.int32),
            "_gram_norms": np.zeros(capacity, dtype=np.float64),
        }
        size = self._size
        for name, array in grown.items():
            if size and self._capacity:
                array[:size] = getattr(self, name)[:size]
        for name, array in grown.items():
            setattr(self, name, array)
        self._capacity = capacity

    def _load(self) -> None:
        rows = self.store.metadata(self.dim)
        highest = max((int(row[0]) for row in rows), default=0)
        self._size = 0
        self._ensure(max(64, len(rows)))
        for row in rows:
            engram_id, ts, label, gram_blob, tier, weight = row[:6]
            namespace = str(row[6]) if len(row) > 6 else "default"
            gram = _decode_gram(gram_blob)
            if gram.size != self.gram_dims:
                # A table can hold rows from another configuration. Such a row
                # is not addressable by this index and must not corrupt it.
                continue
            index = self._size
            self._ids[index] = int(engram_id)
            self._ts[index] = float(ts)
            self._labels[index] = int(label)
            self._grams[index] = gram
            self._gram_norms[index] = float(np.linalg.norm(gram))
            self._tier[index] = int(tier)
            self._weight[index] = float(weight)
            self._ns[index] = self._ns_code(namespace)
            self._size += 1
        self._next_id = highest + 1
        for engram_id in self._ids[: self._size]:
            self._prefetched.add(int(engram_id))
        self.rebuild_charts()

    # ---------------------------------------------------------------- write
    def write(
        self,
        wave: np.ndarray,
        label: int = -1,
        info_gain: float = 0.0,
        tier: int = 1,
        ts: Optional[float] = None,
        namespace: str = "default",
    ) -> Engram:
        """Persist one engram. Returns the stored record.

        Directive F1: the namespace is part of the row, so two tenants can
        hold contradictory engrams under the same numeric label without
        either one being able to retrieve the other.
        """
        namespace = str(namespace or "default")
        wave = _unit(wave)
        gram = (self.projection @ wave).astype(np.complex64)
        stamp = float(time.time() if ts is None else ts)
        with self._lock:
            engram_id = self._next_id
            self._next_id += 1
            self._ensure(self._size + 1)
            index = self._size
            record = Engram(
                engram_id=int(engram_id),
                ts=stamp,
                label=int(label),
                wave=wave.astype(np.complex64),
                gram=gram,
                lyapunov=float(np.real(np.vdot(wave, wave))),
                info_gain=float(info_gain),
                tier=int(tier),
                weight=1.0,
                namespace=namespace,
            )
            self.store.append(record)
            self._ids[index] = engram_id
            self._ts[index] = stamp
            self._labels[index] = int(label)
            self._grams[index] = gram
            self._gram_norms[index] = float(np.linalg.norm(gram))
            self._tier[index] = int(tier)
            self._weight[index] = 1.0
            self._ns[index] = self._ns_code(namespace)
            self._size += 1
            self._chart_insert(index, gram)
            # Holographic superposition: a running boundary wave, normalised.
            self.boundary = _unit(self.boundary + wave)
            key = np.exp(1j * np.angle(wave)).astype(np.complex64)
            self.boundary_key = _unit(self.boundary_key + pbind(key, wave))
            self._put_cache(engram_id, wave)
        return record

    def write_many(self, waves: Iterable[np.ndarray], labels: Iterable[int], tier: int = 1) -> int:
        count = 0
        for wave, label in zip(waves, labels):
            self.write(wave, int(label), tier=tier)
            count += 1
        return count

    # -------------------------------------------------------- chart index (S3)
    def _chart_insert(self, index: int, gram: np.ndarray) -> None:
        """Assign one resident row to a chart, creating one when needed.

        A row joins the chart whose unit centroid is most coherent with it,
        provided the chart still has capacity and the coherence clears
        chart_affinity; otherwise a new chart is opened. Assignment is O(K) in
        the number of charts, never O(N).
        """
        unit = np.asarray(gram, dtype=np.complex64)
        unit = unit / max(float(np.linalg.norm(unit)), _EPS)
        best = -1.0
        best_id = -1
        for chart_id, chart in enumerate(self.charts):
            if int(chart["size"]) >= int(self.chart_capacity):
                continue
            similarity = float(np.real(np.vdot(chart["centroid"], unit)))
            if similarity > best:
                best = similarity
                best_id = chart_id
        if best_id < 0 or best < float(self.chart_affinity):
            chart = {
                "rows": [],
                "sum": np.zeros(self.gram_dims, dtype=np.complex64),
                "centroid": unit.astype(np.complex64),
                "namespace": str(self.namespaces[int(self._ns[index])]),
                "size": 0,
            }
            self.charts.append(chart)
            best_id = len(self.charts) - 1
        chart = self.charts[best_id]
        chart["rows"].append(int(index))
        chart["sum"] = np.asarray(chart["sum"]) + unit
        chart["centroid"] = _unit(np.asarray(chart["sum"])).astype(np.complex64)
        chart["size"] = int(chart["size"]) + 1
        chart["row_array"] = None
        self._chart_of[index] = int(best_id)
        self._centroid_matrix = None

    def rebuild_charts(self) -> Dict[str, int]:
        """Re-partition every resident row. Used after a load or a delete."""
        self.charts = []
        self._chart_of[: self._capacity] = -1
        for index in range(int(self._size)):
            self._chart_insert(index, self._grams[index])
        return self.chart_stats()

    def chart_stats(self) -> Dict[str, object]:
        sizes = [int(chart["size"]) for chart in self.charts]
        return {
            "charts": int(len(self.charts)),
            "chart_capacity": int(self.chart_capacity),
            "engrams": int(self._size),
            "largest_chart": int(max(sizes) if sizes else 0),
            "smallest_chart": int(min(sizes) if sizes else 0),
            "mean_chart": float(np.mean(sizes) if sizes else 0.0),
        }

    def build_centered_index(
        self,
        namespace: Optional[str] = "default",
        *,
        capacity: Optional[int] = None,
        tau_boundary: Optional[float] = None,
        top_charts: int = 4,
        top_charts_max: int = 6,
        calibration_sample: int = 256,
        replication: int = 1,
        eps: float = 1e-3,
    ) -> Dict[str, object]:
        """Fit the T2 mean-centered atlas over one namespace's resident rows.

        The fit reads the CURRENT Gram rows; it writes no wave and changes no
        retrieval arithmetic. A caller can therefore rebuild it, compare it
        against the flat scan, and only then let the automatic policy use it.
        """
        with self._lock:
            size = int(self._size)
            if namespace is None:
                rows = np.arange(size, dtype=np.int64)
            else:
                code = self._ns_index.get(str(namespace))
                if code is None:
                    raise KeyError(f"namespace {namespace!r} has no resident rows")
                rows = np.flatnonzero(self._ns[:size] == int(code)).astype(np.int64)
            if rows.size == 0:
                raise KeyError(f"namespace {namespace!r} has no resident rows")
            grams = np.asarray(self._grams[rows], dtype=np.complex128)
        index = CenteredChartIndex(
            capacity=int(self.chart_capacity if capacity is None else capacity),
            affinity=float(self.chart_affinity),
            tau_boundary=float(0.10 if tau_boundary is None else tau_boundary),
            top_charts=int(top_charts),
            top_charts_max=int(top_charts_max),
            namespace=None if namespace is None else str(namespace),
            replication=int(replication),
            eps=float(eps),
        )
        stats = index.build(grams, rows=rows, namespace=namespace)
        margin = index.calibrate_margin(grams, sample=int(calibration_sample))
        if tau_boundary is not None:
            index.tau_boundary = float(tau_boundary)
            margin["tau_boundary"] = float(tau_boundary)
        with self._lock:
            self.centered_index = index
        return {
            **stats,
            "namespace": None if namespace is None else str(namespace),
            "margin": margin,
            "auto_threshold": int(self.auto_centered_threshold),
        }

    def build_riemannian_atlas(
        self,
        namespace: Optional[str] = "default",
        *,
        rank: int = 8,
        capacity: Optional[int] = None,
        tau_orth: float = 0.55,
        tau_cap: float = 0.35,
        top_charts: int = 2,
        top_charts_max: int = 6,
    ) -> Dict[str, object]:
        """Fit the Sovereign T2 tangent-space atlas over one namespace."""
        from henri.holarch.riemannian_atlas import RiemannianChartAtlas

        with self._lock:
            size = int(self._size)
            if namespace is None:
                rows = np.arange(size, dtype=np.int64)
            else:
                code = self._ns_index.get(str(namespace))
                if code is None:
                    raise KeyError(f"namespace {namespace!r} has no resident rows")
                rows = np.flatnonzero(self._ns[:size] == int(code)).astype(np.int64)
            if rows.size == 0:
                raise KeyError(f"namespace {namespace!r} has no resident rows")
            grams = np.asarray(self._grams[rows], dtype=np.complex128)
        atlas = RiemannianChartAtlas(
            rank=int(rank),
            capacity=int(self.chart_capacity if capacity is None else capacity),
            tau_orth=float(tau_orth),
            tau_cap=float(tau_cap),
            top_charts=int(top_charts),
            top_charts_max=int(top_charts_max),
        )
        stats = atlas.build(grams, rows, namespace=None if namespace is None else str(namespace))
        with self._lock:
            self.riemannian_atlas = atlas
        return {
            **stats,
            "namespace": None if namespace is None else str(namespace),
        }

    def observe_riemannian(
        self, gram: np.ndarray, row: int, namespace: str = "online"
    ) -> Dict[str, object]:
        """Absorb or spawn a chart for one online write."""
        if self.riemannian_atlas is None:
            raise RuntimeError("build_riemannian_atlas must run before observe_riemannian")
        result = self.riemannian_atlas.observe(gram, int(row), namespace=str(namespace))
        if str(result.get("action")) == "spawn":
            self.atlas_online_charts += 1
        return result

    def _centered_for(self, namespace: Optional[str]) -> Optional[CenteredChartIndex]:
        """The atlas that may serve this namespace, or None."""
        index = self.centered_index
        if index is None or not index.synchronized:
            return None
        wanted = None if namespace is None else str(namespace)
        if index.namespace is not None and index.namespace != wanted:
            return None
        return index

    def _chart_selection(
        self, query_gram: np.ndarray, top_charts: int
    ) -> Tuple[np.ndarray, List[int]]:
        """Rows of the winning charts, and the chart ids that won."""
        size = int(self._size)
        if not self.charts or int(top_charts) <= 0:
            return np.arange(size, dtype=np.int64), []
        centroids = self._centroid_matrix
        if centroids is None or int(centroids.shape[0]) != len(self.charts):
            centroids = np.stack([np.asarray(chart["centroid"]) for chart in self.charts])
            centroids = centroids.astype(np.complex64)
            self._centroid_matrix = centroids
        unit = np.asarray(query_gram, dtype=np.complex64)
        unit = unit / max(float(np.linalg.norm(unit)), _EPS)
        coherence = np.real(centroids.conj() @ unit)
        take = int(min(max(int(top_charts), 1), len(self.charts)))
        winners = np.argsort(-coherence, kind="stable")[:take]
        rows: List[np.ndarray] = []
        for chart_id in winners:
            chart = self.charts[int(chart_id)]
            cached = chart.get("row_array")
            if cached is None or int(np.asarray(cached).size) != int(chart["size"]):
                cached = np.asarray(chart["rows"], dtype=np.int64)
                chart["row_array"] = cached
            rows.append(np.asarray(cached, dtype=np.int64))
        selected = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        return selected, [int(chart_id) for chart_id in winners]

    def orthogonalize_chart(
        self,
        chart_id: int,
        tau_crosstalk: float = 0.12,
        eta: float = 0.5,
        rounds: int = 8,
    ) -> Dict[str, object]:
        """Reduce intra-chart Gram crosstalk by a Stiefel retraction (S3).

        The chart's Gram block W is retracted toward its polar factor
        U V^dagger, which is the closest matrix with orthogonal rows in the
        Frobenius sense. Because m rows can exceed d columns, the polar factor
        is only a partial isometry; a bounded number of pairwise deflations on
        the most coherent pair then finishes the job. The waves themselves are
        untouched - only the chart's routing projection moves - so the exact
        coherence rescoring at egress still ranks the same neighbours.
        """
        if not 0 <= int(chart_id) < len(self.charts):
            raise KeyError(f"chart {chart_id} does not exist")
        chart = self.charts[int(chart_id)]
        rows = np.asarray(chart["rows"], dtype=np.int64)
        if rows.size < 2:
            return {"chart": int(chart_id), "rows": int(rows.size), "applied": False}
        gram = np.asarray(self._grams[rows], dtype=np.complex128)
        norms = np.linalg.norm(gram, axis=1)
        keep = norms > _EPS
        rows = rows[keep]
        gram = gram[keep]
        norms = norms[keep]
        if rows.size < 2:
            return {"chart": int(chart_id), "rows": int(rows.size), "applied": False}
        unit = gram / norms[:, None]

        def crosstalk(matrix: np.ndarray) -> float:
            squared = np.abs(matrix @ matrix.conj().T) ** 2
            np.fill_diagonal(squared, 0.0)
            return float(np.sqrt(squared.max())) if squared.size else 0.0

        before = crosstalk(unit)
        if before <= float(tau_crosstalk):
            return {
                "chart": int(chart_id),
                "rows": int(rows.size),
                "applied": False,
                "crosstalk_before": float(before),
                "crosstalk_after": float(before),
                "tau_crosstalk": float(tau_crosstalk),
            }
        left, singular, right = np.linalg.svd(gram, full_matrices=False)
        polar = left @ right
        retracted = (1.0 - float(eta)) * gram + float(eta) * polar
        retracted = retracted * (norms[:, None] / np.maximum(
            np.linalg.norm(retracted, axis=1)[:, None], _EPS
        ))
        unit = retracted / np.maximum(np.linalg.norm(retracted, axis=1)[:, None], _EPS)
        for _ in range(max(int(rounds), 0)):
            coherence = np.abs(unit @ unit.conj().T) ** 2
            np.fill_diagonal(coherence, 0.0)
            pair = np.unravel_index(int(np.argmax(coherence)), coherence.shape)
            i, j = int(pair[0]), int(pair[1])
            overlap = np.vdot(unit[i], unit[j])
            if abs(overlap) <= float(tau_crosstalk):
                break
            unit[j] = unit[j] - overlap * unit[i]
            unit[j] = unit[j] / max(float(np.linalg.norm(unit[j])), _EPS)
            retracted[j] = unit[j] * float(norms[j])
        after = crosstalk(unit)
        with self._lock:
            self._grams[rows] = retracted.astype(np.complex64)
            chart["sum"] = np.asarray(unit.sum(axis=0), dtype=np.complex64)
            chart["centroid"] = _unit(np.asarray(chart["sum"])).astype(np.complex64)
            for row, engram_id in zip(rows.tolist(), self._ids[rows].tolist()):
                wave = None
                cached = self._cache.get(int(engram_id))
                if cached is not None:
                    wave = cached
                if wave is None:
                    continue
                try:
                    self.store.replace_wave(
                        int(engram_id), wave, self._grams[int(row)]
                    )
                except Exception:
                    pass
        return {
            "chart": int(chart_id),
            "rows": int(rows.size),
            "applied": True,
            "crosstalk_before": float(before),
            "crosstalk_after": float(after),
            "reduced": bool(after < before),
            "tau_crosstalk": float(tau_crosstalk),
            "eta": float(eta),
        }

    # ----------------------------------------------------------------- read
    def _put_cache(self, engram_id: int, wave: np.ndarray) -> None:
        self._cache[engram_id] = wave
        self._cache.move_to_end(engram_id)
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)

    def _fetch(self, ids: Sequence[int]) -> List[np.ndarray]:
        """Return waves for ids, in order, using the LRU cache as a side effect.

        The return value is assembled from a local map, not read back out of
        the cache. A candidate set larger than the cache capacity would evict
        its own earlier members between the store fetch and the read.
        """
        ids = [int(i) for i in ids]
        # Directive N2: the resident LRU snapshot is read under the short lock,
        # then released. A miss pages the store on the store's own connection
        # lock, not on the memory lock, so sixteen readers do not queue behind
        # one disk read. The cache is re-acquired only to insert the fetched
        # rows and bump the two counters.
        with self._lock:
            resolved: Dict[int, np.ndarray] = {
                engram_id: self._cache[engram_id]
                for engram_id in ids
                if engram_id in self._cache
            }
        missing = [i for i in ids if i not in resolved]
        if missing:
            fetched = list(self.store.fetch(missing))
            for record in fetched:
                resolved[int(record.engram_id)] = record.wave
            with self._lock:
                for record in fetched:
                    self._put_cache(record.engram_id, record.wave)
                self._cache_misses += len(missing)
                if len(missing) < len(ids):
                    self._cache_hits += len(ids) - len(missing)
        else:
            with self._lock:
                self._cache_hits += len(ids)
        return [resolved[int(i)] for i in ids if int(i) in resolved]

    def query(
        self,
        wave: np.ndarray,
        k: int = 8,
        avoid_ids: Sequence[int] = (),
        candidate_multiplier: int = 6,
        task: Optional[str] = None,
        namespace: Optional[str] = "default",
        use_charts: Optional[bool] = None,
        top_charts: Optional[int] = None,
        use_riemannian: bool = False,
    ) -> MemoryRead:
        """Retrieve the k best engrams and fuse them into one context wave.

        Decision D1: when task is given, only engrams whose stored label lies
        in that task's disjoint label block are candidates. A task-scoped
        query with no candidate in its block returns an empty read, which the
        C1 rule turns into a veto for inference.

        Directive F1: when namespace is given (the default is "default", not
        None), only that tenant's rows are candidates. A namespace with no
        row therefore reads as an empty, fail-closed result rather than
        borrowing another tenant's memory.
        """
        start = time.perf_counter()
        query_wave = _unit(wave)
        query_gram = self.projection @ query_wave
        query_norm = float(np.linalg.norm(query_gram))
        wave_norm = float(np.linalg.norm(query_wave))
        # Directive M2: capture an immutable snapshot under the lock, then
        # release it. A writer appends at index size or reallocates the arrays;
        # both leave this captured view intact, so the matrix-vector scoring
        # and the coherence sort run outside the mutex.
        with self._lock:
            size = self._size
            if size == 0:
                self._last_query = query_wave
                return MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0,
                                  (time.perf_counter() - start) * 1e3, 0, True, 0.0, None)
            grams_all = self._grams
            ids_all = self._ids
            labels_all = self._labels
            ns_all = self._ns
            ts_all = self._ts
            tier_all = self._tier
            ns_code = self._ns_index.get(str(namespace)) if namespace is not None else None
        # Directive S3: two-tier chart routing. The query scores the K chart
        # centroids, then rescans only the winning charts' rows. With
        # use_charts=False the scan is the flat O(N) reference, so the two arms
        # of the gate are directly comparable on the same query.
        winners: List[int] = []
        centered_used = False
        atlas_used = False
        routing_margin = 0.0
        centered_index = self._centered_for(namespace)
        if bool(use_riemannian) and self.riemannian_atlas is not None:
            selected, routing = self.riemannian_atlas.select(
                query_gram,
                namespace=None if namespace is None else str(namespace),
                top_charts=top_charts,
            )
            self.chart_queries += 1
            self.chart_scanned += int(selected.size)
            winners = [int(value) for value in routing.get("charts") or []]
            routing_margin = float(routing.get("margin", 0.0))
            if selected.size:
                row_index = selected
                atlas_used = True
            else:
                row_index = np.arange(size, dtype=np.int64)
        if use_charts is None:
            # Directive T2: automatic only when a synchronized centered atlas
            # exists for this namespace AND the namespace holds at least 1024
            # rows. Otherwise the historical flat read is unchanged.
            use_charts = bool(
                centered_index is not None
                and int(centered_index.size) >= int(self.auto_centered_threshold)
            )
        if atlas_used:
            pass
        elif bool(use_charts) and centered_index is not None:
            selected, routing = centered_index.select(query_gram, top_charts=top_charts)
            self.chart_queries += 1
            self.chart_scanned += int(selected.size)
            winners = [int(value) for value in routing.get("charts") or []]
            routing_margin = float(routing.get("margin", 0.0))
            if selected.size:
                row_index = selected
                centered_used = True
            else:
                row_index = np.arange(size, dtype=np.int64)
        elif bool(use_charts) and self.charts:
            take = int(self.top_charts_default if top_charts is None else top_charts)
            selected, winners = self._chart_selection(query_gram, take)
            self.chart_queries += 1
            self.chart_scanned += int(selected.size)
            row_index = selected if selected.size else np.arange(size, dtype=np.int64)
        else:
            row_index = np.arange(size, dtype=np.int64)
            self.flat_scanned += int(size)
        grams = grams_all[row_index]
        ids = ids_all[row_index]
        labels = labels_all[row_index]
        ns_vec = ns_all[row_index]
        ts_vec = ts_all[row_index]
        tier_vec = tier_all[row_index]
        size = int(row_index.size)
        # The batched norm, not a per-row cached one: the two differ in the
        # last ulp, and a cartridge selection or a Cul-de-Sac trap can turn on
        # a near tie. The deployed arithmetic is preserved bit for bit.
        norms = np.linalg.norm(grams, axis=1)
        # The arithmetic is the deployed one: divide by (norms * norm(query)),
        # in that order. A reciprocal-multiply rewrites the last bits of every
        # score, and a Cul-de-Sac trap is decided by a near tie.
        scores = np.real(grams @ np.conj(query_gram)) / np.maximum(
            norms * (query_norm + _EPS), _EPS
        )
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        if namespace is not None:
            allowed_ns = (
                np.zeros(size, dtype=bool)
                if ns_code is None
                else (ns_vec == int(ns_code))
            )
            if not bool(allowed_ns.any()):
                self._last_query = query_wave
                return MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0,
                                  (time.perf_counter() - start) * 1e3, 0, True, 0.0, None)
            scores = np.where(allowed_ns, scores, -np.inf)
        if task is not None:
            lo, hi = task_label_block(task)
            allowed = (labels >= lo) & (labels < hi)
            if not bool(allowed.any()):
                self._last_query = query_wave
                return MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0,
                                  (time.perf_counter() - start) * 1e3, 0, True, 0.0, None)
            scores = np.where(allowed, scores, -np.inf)
        if avoid_ids:
            blocked = np.isin(ids, np.asarray(list(avoid_ids), dtype=np.int64))
            scores = np.where(blocked, -np.inf, scores)
        if not bool(np.isfinite(scores).any()):
            self._last_query = query_wave
            return MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0,
                              (time.perf_counter() - start) * 1e3, 0, True, 0.0, None)
        s_max = float(scores.max())
        finite = np.flatnonzero(np.isfinite(scores))
        if finite.size == 0:
            self._last_query = query_wave
            return MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0,
                              (time.perf_counter() - start) * 1e3, 0, True, 0.0, None)
        pool = int(min(finite.size, max(int(k) * int(candidate_multiplier), 32)))
        # Decision D1: rank only the candidates that survived the task
        # and avoid filters. Ranking the full index and truncating later
        # would re-admit a filtered engram by its raw coherence.
        local = np.argpartition(-scores[finite], pool - 1)[:pool]
        candidates = finite[local]
        # Constraint C11: never rank a tie plateau with a bare argmax.
        # Sort by (score descending, engram id ascending), which is a
        # total order even when scores are exactly equal.
        order = sorted(
            candidates.tolist(),
            key=lambda i: (-float(scores[i]), int(ids[i])),
        )
        candidate_ids = [int(ids[i]) for i in order]
        candidate_scores = {int(ids[i]): float(scores[i]) for i in order}
        candidate_meta = {
            int(ids[i]): (int(labels[i]), float(ts_vec[i]), int(tier_vec[i]))
            for i in order
        }
        waves = self._fetch(candidate_ids)
        by_id = dict(zip(candidate_ids, waves))
        # Directive N2: one vectorised exact-coherence product over the
        # candidate pool replaces the per-candidate Python loop. The
        # arithmetic is the same normalised Hermitian inner product, but it
        # runs as one C contraction, so the read path spends its time in
        # NumPy rather than in the interpreter.
        keep_ids = [engram_id for engram_id in candidate_ids if engram_id in by_id]
        if keep_ids:
            # complex64 keeps the rescoring product on the same precision as
            # the stored waves and halves the bytes the egress path moves.
            candidate_matrix = np.stack([by_id[engram_id] for engram_id in keep_ids]).astype(
                np.complex128
            )
            query_vector = np.asarray(query_wave, dtype=np.complex128).reshape(-1)
            # The wave norm was already computed at ingress. Recomputing it
            # here was a second D-length norm on every read; the arithmetic is
            # the same normalised Hermitian product.
            denominator = np.linalg.norm(candidate_matrix, axis=1) * (
                wave_norm + _EPS
            )
            exact_scores = np.real(candidate_matrix @ np.conj(query_vector)) / denominator
            exact = [
                (int(engram_id), float(score))
                for engram_id, score in zip(keep_ids, exact_scores)
            ]
        else:
            exact = []
        exact.sort(key=lambda pair: (-pair[1], pair[0]))
        top = exact[: int(k)]
        hits = [
            MemoryHit(
                engram_id=engram_id,
                label=candidate_meta.get(engram_id, (-1, 0.0, 1))[0],
                score=float(candidate_scores.get(engram_id, 0.0)),
                coherence=float(coh),
                ts=candidate_meta.get(engram_id, (-1, 0.0, 1))[1],
                tier=candidate_meta.get(engram_id, (-1, 0.0, 1))[2],
            )
            for engram_id, coh in top
        ]
        weights = np.array([max(hit.coherence, 0.0) for hit in hits], dtype=np.float64)
        if weights.sum() > _EPS:
            weights = weights / weights.sum()
            context = np.zeros(self.dim, dtype=np.complex64)
            for weight, hit in zip(weights, hits):
                # Use the local fetch result, not the LRU cache: a fetch with
                # more candidates than the cache capacity would evict an entry
                # between scoring and fusion.
                context = context + np.float32(weight) * by_id[hit.engram_id]
            context = _unit(context)
        else:
            context = np.zeros(self.dim, dtype=np.complex64)

        elapsed = (time.perf_counter() - start) * 1e3
        with self._lock:
            self._reads += 1
            self._read_ms_total += elapsed
            if self._last_query is not None:
                self._last_query = query_wave
            else:
                self._last_query = query_wave
        return MemoryRead(
            hits=hits,
            context=context,
            context_norm=float(np.linalg.norm(context)),
            latency_ms=elapsed,
            candidates=len(candidate_ids),
            used_boundary=False,
            s_max=float(s_max),
            top_wave=(by_id[top[0][0]] if top else None),
            scanned=int(row_index.size),
            probed_charts=int(len(winners)),
            centered=bool(centered_used),
            atlas=bool(atlas_used),
            routing_margin=float(routing_margin),
        )

    def fetch_wave(self, engram_id: int) -> np.ndarray:
        """Return one stored wave by id, paging it from disk when needed."""
        waves = self._fetch([int(engram_id)])
        if not waves:
            raise KeyError(f"engram {engram_id} not found")
        return waves[0]

    def _index_of(self, engram_id: int) -> int:
        matches = np.flatnonzero(self._ids[: self._size] == int(engram_id))
        return int(matches[0]) if matches.size else 0

    # --------------------------------------------------------------- context
    def query_batch(
        self,
        waves,
        k: int = 8,
        task: Optional[str] = None,
        namespace: Optional[str] = "default",
    ):
        """Score a batch of query waves with one Gram matrix product.

        The projection and the coherence ranking are one (B, gram_dims) x
        (gram_dims, N) product, so the per-row Python loop is reduced to the
        exact-wave fetch and fusion. The snapshot is captured under the lock
        and scored outside it, the same rule query() follows.
        """
        matrix = np.asarray(waves, dtype=np.complex64)
        if matrix.ndim != 2:
            raise ValueError(f"expected (batch, dim) waves, got {matrix.shape}")
        batch = int(matrix.shape[0])
        unit_rows = np.stack([_unit(row) for row in matrix]).astype(np.complex64)
        query_grams = (unit_rows @ self.projection.T).astype(np.complex64)
        with self._lock:
            size = self._size
            if size == 0:
                return [
                    MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0, 0.0, 0, True, 0.0, None)
                    for _ in range(batch)
                ]
            grams_all = self._grams
            ids_all = self._ids
            labels_all = self._labels
            ns_all = self._ns
            ts_all = self._ts
            tier_all = self._tier
            ns_code = self._ns_index.get(str(namespace)) if namespace is not None else None
        grams = grams_all[:size]
        ids = ids_all[:size]
        labels = labels_all[:size]
        norms = np.linalg.norm(grams, axis=1)
        gram_norms = np.linalg.norm(query_grams, axis=1)
        scores_matrix = np.real(query_grams @ np.conj(grams.T)) / np.maximum(
            gram_norms[:, None] * norms[None, :], _EPS
        )
        scores_matrix = np.nan_to_num(scores_matrix, nan=-1.0, posinf=-1.0, neginf=-1.0)
        allowed = np.ones(size, dtype=bool)
        if namespace is not None:
            if ns_code is None:
                allowed[:] = False
            else:
                allowed &= ns_all[:size] == int(ns_code)
        if task is not None:
            lo, hi = task_label_block(task)
            allowed &= (labels >= lo) & (labels < hi)
        scores_matrix = np.where(allowed[None, :], scores_matrix, -np.inf)
        reads = []
        for row in range(batch):
            scores = scores_matrix[row]
            finite = np.flatnonzero(np.isfinite(scores))
            if finite.size == 0:
                reads.append(
                    MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0, 0.0, 0, True, 0.0, None)
                )
                continue
            s_max = float(scores.max())
            pool = int(min(finite.size, max(int(k) * 6, 32)))
            local = np.argpartition(-scores[finite], pool - 1)[:pool]
            candidates = finite[local]
            order = sorted(
                candidates.tolist(), key=lambda i: (-float(scores[i]), int(ids[i]))
            )
            candidate_ids = [int(ids[i]) for i in order]
            candidate_scores = {int(ids[i]): float(scores[i]) for i in order}
            candidate_meta = {
                int(ids[i]): (int(labels[i]), float(ts_all[i]), int(tier_all[i]))
                for i in order
            }
            waves_fetched = self._fetch(candidate_ids)
            by_id = dict(zip(candidate_ids, waves_fetched))
            exact = [
                (engram_id, coherence(unit_rows[row], by_id[engram_id]))
                for engram_id in candidate_ids
                if engram_id in by_id
            ]
            exact.sort(key=lambda pair: (-pair[1], pair[0]))
            top = exact[: int(k)]
            hits = [
                MemoryHit(
                    engram_id=engram_id,
                    label=candidate_meta.get(engram_id, (-1, 0.0, 1))[0],
                    score=float(candidate_scores.get(engram_id, 0.0)),
                    coherence=float(coh),
                    ts=candidate_meta.get(engram_id, (-1, 0.0, 1))[1],
                    tier=candidate_meta.get(engram_id, (-1, 0.0, 1))[2],
                )
                for engram_id, coh in top
            ]
            weights = np.array([max(hit.coherence, 0.0) for hit in hits], dtype=np.float64)
            if weights.sum() > _EPS:
                weights = weights / weights.sum()
                context = np.zeros(self.dim, dtype=np.complex64)
                for weight, hit in zip(weights, hits):
                    context = context + np.float32(weight) * by_id[hit.engram_id]
                context = _unit(context)
            else:
                context = np.zeros(self.dim, dtype=np.complex64)
            reads.append(
                MemoryRead(
                    hits=hits,
                    context=context,
                    context_norm=float(np.linalg.norm(context)),
                    latency_ms=0.0,
                    candidates=len(candidate_ids),
                    used_boundary=False,
                    s_max=float(s_max),
                    top_wave=(by_id[top[0][0]] if top else None),
                )
            )
        return reads

    def context_tensor_batch(
        self,
        waves,
        k: int = 8,
        task: Optional[str] = None,
        namespace: Optional[str] = "default",
    ):
        """Fixed (B, context_tokens, context_dim) tensor for a batch of waves."""
        reads = self.query_batch(waves, k=k, task=task, namespace=namespace)
        tokens = int(self.cfg.context_tokens)
        prefix = np.zeros(
            (len(reads), tokens, int(self.cfg.context_dim)), dtype=np.float32
        )
        for index, read in enumerate(reads):
            if read.hits:
                chunks = read.context.reshape(tokens, self.cfg.chunk)
                prefix[index] = (
                    chunks.real @ self.context_proj + chunks.imag @ self.context_proj
                ) / np.sqrt(2.0)
        return prefix, reads

    def context_tensor(
        self,
        wave: np.ndarray,
        k: int = 8,
        task: Optional[str] = None,
        namespace: Optional[str] = "default",
    ) -> Tuple[np.ndarray, MemoryRead]:
        """Fixed (context_tokens, context_dim) tensor at every store size."""
        read = self.query(wave, k=k, task=task, namespace=namespace)
        tokens = int(self.cfg.context_tokens)
        if read.hits:
            chunks = read.context.reshape(tokens, self.cfg.chunk)
            prefix = (
                chunks.real @ self.context_proj + chunks.imag @ self.context_proj
            ) / np.sqrt(2.0)
        else:
            prefix = np.zeros((tokens, int(self.cfg.context_dim)), dtype=np.float32)
        return prefix.astype(np.float32), read

    def axiomatic_engrams(self, namespace: Optional[str] = None) -> List[Tuple[int, int]]:
        """(engram_id, label) for every axiomatic (tier-3) row.

        Tier-3 rows are the clean prototype bank of Decision 2. They are
        never pruned, so the attractor bank survives compaction and decay.
        """
        rows = self.store.metadata(self.dim, namespace=namespace)
        return [(int(row[0]), int(row[2])) for row in rows if int(row[4]) >= TIER_AXIOMATIC]

    # -------------------------------------------------------------- metadata
    def put_meta(self, namespace: str, kind: str, key: str, payload: dict) -> bool:
        """Persist one metadata record for a tenant (Directive G4)."""
        putter = getattr(self.store, "put_meta", None)
        if putter is None:
            return False
        putter(str(namespace), str(kind), str(key), dict(payload))
        return True

    def get_meta(self, namespace: str, kind: str, key: Optional[str] = None) -> List[Tuple[str, dict]]:
        getter = getattr(self.store, "get_meta", None)
        if getter is None:
            return []
        return list(getter(str(namespace), str(kind), key))

    def meta_namespaces(self) -> List[str]:
        getter = getattr(self.store, "meta_namespaces", None)
        if getter is None:
            return []
        return list(getter())

    # ------------------------------------------------------- global epoch (O2)
    STATE_EPOCH_NAMESPACE = "__system__"
    STATE_EPOCH_KIND = "sync"
    STATE_EPOCH_KEY = "state_epoch"

    def state_epoch(self) -> int:
        """Read the database-backed mutation epoch shared by every lane.

        Directive O2. The value lives in engram_meta, so eight SidecarService
        processes that open the same store see one monotonic counter. A read
        of an absent table returns 0, which is the pre-Cycle-O state.
        """
        rows = self.get_meta(self.STATE_EPOCH_NAMESPACE, self.STATE_EPOCH_KIND, self.STATE_EPOCH_KEY)
        if not rows:
            return 0
        try:
            return int(dict(rows[0][1]).get("epoch", 0))
        except Exception:
            return 0

    def bump_state_epoch(self) -> int:
        """Advance the shared epoch and return the new value.

        The read-modify-write is serialised by the memory lock inside one
        process and by SQLite's own write lock across processes. The epoch is
        metadata, not an engram, so it changes no retrieval result.
        """
        with self._lock:
            epoch = int(self.state_epoch()) + 1
            self.put_meta(
                self.STATE_EPOCH_NAMESPACE,
                self.STATE_EPOCH_KIND,
                self.STATE_EPOCH_KEY,
                {"epoch": epoch, "updated_ts": float(time.time())},
            )
            return epoch

    # ------------------------------------------------------- boundary axioms
    def register_axiom(
        self,
        wave: np.ndarray,
        name: str,
        polarity: str,
        source: str = "human_heuristic",
        slot_mask: Sequence[bool] = (False, False, False, True),
        namespace: str = "default",
        key: Optional[str] = None,
    ) -> BoundaryAxiom:
        """Persist one typed boundary axiom as a tier-3 engram (Directive K2).

        The wave is masked to the governed slots before it is written, so a
        Slot-3 grammar rule cannot accidentally steer Slot 0 geometry. The
        metadata row is the registry entry; the engram itself is the wave the
        Zone B relaxation reads back. A repeated key appends a new row and
        updates the registry pointer, which is how test-time learning replaces
        a refuted axiom without deleting history.
        """
        polarity_value = str(polarity).upper()
        if polarity_value not in BOUNDARY_POLARITIES:
            raise ValueError(f"polarity must be one of {BOUNDARY_POLARITIES}, got {polarity!r}")
        source_value = str(source)
        if source_value not in BOUNDARY_SOURCES:
            raise ValueError(f"source must be one of {BOUNDARY_SOURCES}, got {source!r}")
        mask = tuple(bool(value) for value in slot_mask)
        if len(mask) != 4:
            raise ValueError("slot_mask must hold four booleans")
        if not any(mask):
            raise ValueError("an axiom must govern at least one slot")
        base = _unit(wave)
        slots = base.reshape(4, -1).copy()
        slots[[index for index, keep in enumerate(mask) if not keep]] = 0.0
        governed = _unit(slots.reshape(-1))
        axiom_id = str(key or f"{namespace}:{polarity_value}:{name}")
        digest = hashlib.sha256(axiom_id.encode("utf-8")).hexdigest()
        label = AXIOM_LABEL_BASE + (int(digest[:8], 16) % 900)
        record = self.write(
            governed, label=int(label), tier=TIER_AXIOMATIC, namespace=namespace
        )
        axiom = BoundaryAxiom(
            axiom_id=axiom_id,
            name=str(name),
            polarity=polarity_value,
            source=source_value,
            slot_mask=mask,
            namespace=str(namespace or "default"),
            engram_id=int(record.engram_id),
            wave=governed,
            created_ts=float(record.ts),
        )
        self.put_meta(axiom.namespace, BOUNDARY_AXIOM_KIND, axiom.axiom_id, axiom.as_payload())
        return axiom

    def boundary_axioms(
        self,
        namespace: str = "default",
        polarity: Optional[str] = None,
    ) -> List[BoundaryAxiom]:
        """Every registered axiom of one tenant, optionally of one polarity."""
        wanted = None if polarity is None else str(polarity).upper()
        out: List[BoundaryAxiom] = []
        for key, payload in self.get_meta(namespace, BOUNDARY_AXIOM_KIND):
            value = str(payload.get("polarity", ""))
            if wanted is not None and value != wanted:
                continue
            engram_id = int(payload.get("engram_id", -1))
            try:
                wave = self.fetch_wave(engram_id)
            except (KeyError, ValueError):
                continue
            out.append(
                BoundaryAxiom(
                    axiom_id=str(payload.get("axiom_id", key)),
                    name=str(payload.get("name", "")),
                    polarity=value,
                    source=str(payload.get("source", "")),
                    slot_mask=tuple(bool(v) for v in payload.get("slot_mask", (0, 0, 0, 1))),
                    namespace=str(payload.get("namespace", namespace)),
                    engram_id=engram_id,
                    wave=np.asarray(wave, dtype=np.complex64),
                    created_ts=float(payload.get("created_ts", 0.0)),
                )
            )
        out.sort(key=lambda axiom: (axiom.engram_id, axiom.axiom_id))
        return out

    def axiom_waves(
        self,
        namespace: str = "default",
        polarity: Optional[str] = None,
    ) -> np.ndarray:
        """The axioms of one polarity as an (M, dim) complex64 matrix."""
        axioms = self.boundary_axioms(namespace, polarity)
        if not axioms:
            return np.zeros((0, self.dim), dtype=np.complex64)
        return np.stack([axiom.governed() for axiom in axioms]).astype(np.complex64)

    # ------------------------------------------------------------- lifecycle
    def _raw_weight(self, now: Optional[float] = None) -> np.ndarray:
        """Finite-timescale decay, before the axiomatic floor is applied."""
        stamp = float(time.time() if now is None else now)
        with self._lock:
            age = np.maximum(stamp - self._ts[: self._size], 0.0)
            weight = np.zeros(self._size, dtype=np.float64)
            for tau, gain in zip(self.cfg.tau_seconds, self.cfg.tau_weights):
                if math.isinf(float(tau)):
                    # The infinite timescale is the axiomatic reservoir. It is
                    # applied to tier-3 engrams below, not to working memory.
                    continue
                weight += float(gain) * np.exp(-age / float(tau))
            return weight

    def decay(self, now: Optional[float] = None) -> np.ndarray:
        """Viscoelastic retention weight for every engram."""
        weight = self._raw_weight(now)
        with self._lock:
            axiomatic = self._tier[: self._size] >= TIER_AXIOMATIC
            weight[axiomatic] = 1.0
            self._weight[: self._size] = weight.astype(np.float32)
            return weight

    def prune(self, now: Optional[float] = None, min_weight: Optional[float] = None) -> Dict[str, int]:
        """Drop dead working engrams. Axiomatic engrams are never pruned."""
        threshold = float(self.cfg.prune_weight if min_weight is None else min_weight)
        weight = self.decay(now)
        raw = self._raw_weight(now)
        with self._lock:
            tier = self._tier[: self._size]
            dead = (weight < threshold) & (tier < TIER_AXIOMATIC)
            dead_ids = [int(i) for i in self._ids[: self._size][dead]]
            # Count the axioms the decay would have killed had the floor not
            # held. That is the measured meaning of "protected", not a count
            # of every axiomatic row.
            protected = int(np.sum((raw < threshold) & (tier >= TIER_AXIOMATIC)))
            if dead_ids:
                self.store.delete(dead_ids)
                keep = ~dead
                self._compact(keep)
            return {"pruned": len(dead_ids), "protected_axiomatic": protected}

    def _compact(self, keep: np.ndarray) -> None:
        size = self._size
        for name in ("_grams", "_ids", "_ts", "_tier", "_weight", "_labels"):
            value = getattr(self, name)
            value[: int(keep.sum())] = value[:size][keep]
        self._size = int(keep.sum())

    def consolidate(self, chart_size: Optional[int] = None) -> Dict[str, float | int]:
        """Stiefel-retract the most recent chart to lower Gram crosstalk."""
        limit = int(self.cfg.chart_capacity if chart_size is None else chart_size)
        with self._lock:
            size = min(self._size, limit)
            if size < 3:
                return {"chart": size, "crosstalk_before": 0.0, "crosstalk_after": 0.0, "rewritten": 0}
            ids = self._ids[self._size - size : self._size].copy()
        waves = np.stack(self._fetch([int(i) for i in ids]))
        gram = waves @ np.conj(waves.T)
        off = np.abs(gram) - np.eye(size)
        before = float(np.max(off))

        u, s, vh = np.linalg.svd(waves, full_matrices=False)
        retracted = (u @ vh).astype(np.complex64)
        after_gram = retracted @ np.conj(retracted.T)
        after = float(np.max(np.abs(after_gram) - np.eye(size)))
        for index, engram_id in enumerate(ids):
            wave = _unit(retracted[index])
            gram_vec = (self.projection @ wave).astype(np.complex64)
            self.store.replace_wave(int(engram_id), wave, gram_vec)
            with self._lock:
                slot = self._index_of(int(engram_id))
                self._grams[slot] = gram_vec
                self._put_cache(int(engram_id), wave)
        return {
            "chart": size,
            "crosstalk_before": before,
            "crosstalk_after": after,
            "rewritten": int(size),
        }

    def prefetch(self, wave: np.ndarray, k: int = 8) -> Dict[str, int]:
        """Predict the next query from wave velocity and warm the cache."""
        current = _unit(wave)
        with self._lock:
            velocity = (
                current - self._last_query
                if self._last_query is not None and self._last_query.shape == current.shape
                else np.zeros_like(current)
            )
        predicted = _unit(current + 0.5 * velocity)
        read = self.query(predicted, k=k)
        with self._lock:
            self._prefetched |= {hit.engram_id for hit in read.hits}
        return {"prefetched": len(read.hits), "store": self._size}

    def reset(self) -> None:
        """Erase every engram and rebuild an empty index. Destructive."""
        with self._lock:
            self.store.reset()
            self._size = 0
            self._next_id = 1
            self._cache.clear()
            self._prefetched.clear()
            self._last_query = None
            self.boundary = np.zeros(self.dim, dtype=np.complex64)
            self.boundary_key = np.zeros(self.dim, dtype=np.complex64)

    def stats(self) -> Dict[str, float]:
        with self._lock:
            reads = max(self._reads, 1)
            return {
                "backend": self.backend,
                "count": float(self._size),
                "gram_bytes": float(self._size * self.gram_dims * 8),
                "wave_bytes": float(self._size * self.dim * 8),
                "cache_capacity": float(self.cache_capacity),
                "cache_resident": float(len(self._cache)),
                "cache_hit_rate": float(self._cache_hits / max(self._cache_hits + self._cache_misses, 1)),
                "mean_read_ms": float(self._read_ms_total / reads),
                "reads": float(self._reads),
            }

    def close(self) -> None:
        self.store.close()


class PrototypeBank:
    """Clean post-photonic Hopfield attractors, stored in Zone C.

    Evidence label: DERIVED for the energy, OBSERVED for every alignment,
    snap accuracy and beta the calibration returns.

    Decision 2 replaces a pull toward B_theta(psi_context_in) with a pull
    toward sharp prototypes. The reason is physical: the context bundle is
    retrieved with the NOISY query, so it already contains the noise that
    the loop is supposed to remove. A prototype is the unit-normalised mean
    of the post-photonic wave of the CLEAN training samples of one class,
    so the attractor carries no sensor noise by construction.

    The competition is a softmax over alignments with inverse temperature
    beta. Constraint C3 refuses the transferred literal beta* = 26.10: that
    value was fitted on the functor-score readout, and no measurement ties
    it to this one. beta is therefore calibrated from the bank's own
    alignment gaps, and the refused literal is recorded, not used.
    """

    BETA_STAR_REFERENCE = 26.10
    """The C3-refused literal. Kept for audit, never applied."""

    BETA_TARGET_DECADES = 4.0
    """Declared sharpness target: a one-median-gap lead gives e^4 ~ 54.6."""

    def __init__(self, dim: int, beta: float = 0.0) -> None:
        self.dim = int(dim)
        self.phi = np.zeros((0, self.dim), dtype=np.complex64)
        self.labels = np.zeros(0, dtype=np.int64)
        self.counts = np.zeros(0, dtype=np.int64)
        self.beta = float(beta)
        self.beta_source = "declared" if float(beta) > 0.0 else "uncalibrated"

    def __len__(self) -> int:
        return int(self.phi.shape[0])

    # ----------------------------------------------------------------- build
    def build(
        self,
        waves: np.ndarray,
        labels: np.ndarray,
        max_prototypes: int = 64,
    ) -> Dict[str, float]:
        """Build one prototype per class from clean post-photonic waves."""
        waves = np.asarray(waves, dtype=np.complex64)
        labels = np.asarray(labels, dtype=np.int64)
        if waves.ndim != 2 or waves.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}) waves, got {waves.shape}")
        if waves.shape[0] != labels.shape[0]:
            raise ValueError("waves and labels must align")
        rows: List[np.ndarray] = []
        classes: List[int] = []
        counts: List[int] = []
        for label in sorted(int(v) for v in np.unique(labels)):
            block = waves[labels == label]
            if block.size == 0:
                continue
            rows.append(_unit(block.mean(axis=0).astype(np.complex64)))
            classes.append(int(label))
            counts.append(int(block.shape[0]))
        order = np.argsort(-np.asarray(counts))[: int(max_prototypes)]
        self.phi = np.stack([rows[int(i)] for i in order]).astype(np.complex64)
        self.labels = np.asarray([classes[int(i)] for i in order], dtype=np.int64)
        self.counts = np.asarray([counts[int(i)] for i in order], dtype=np.int64)
        return {
            "prototypes": float(len(self.phi)),
            "prototype_classes": float(len(np.unique(self.labels))) if len(self) else 0.0,
            "prototype_min_support": float(self.counts.min()) if len(self) else 0.0,
        }

    # ------------------------------------------------------------- retrieval
    def alignments(self, waves: np.ndarray) -> np.ndarray:
        """Re<psi, phi_m> for every wave, shape (N, M).

        The prototype rows are unit norm and the query is not assumed to
        be, so a caller that needs a cosine must normalise first. The
        energy uses this raw inner product, exactly as written in
        Decision 2.
        """
        waves = np.asarray(waves, dtype=np.complex64)
        if len(self) == 0:
            return np.zeros((waves.shape[0], 0), dtype=np.float32)
        return np.real(waves @ self.phi.conj().T).astype(np.float32)

    def competition(self, wave: np.ndarray) -> Tuple[np.ndarray, int]:
        """Softmax over prototypes at the calibrated beta, and the snap index."""
        scores = self.alignments(np.asarray(wave, dtype=np.complex64).reshape(1, -1))[0]
        if scores.size == 0:
            return np.zeros(0, dtype=np.float32), -1
        scaled = float(self.beta) * scores.astype(np.float64)
        scaled -= scaled.max()
        weights = np.exp(scaled)
        weights /= max(weights.sum(), _EPS)
        return weights.astype(np.float32), int(np.argmax(scores))

    def snap(self, waves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest prototype index and its stored label, shape (N,)."""
        scores = self.alignments(waves)
        if scores.shape[1] == 0:
            return np.full(scores.shape[0], -1, dtype=np.int64), np.zeros(scores.shape[0])
        index = np.argmax(scores, axis=1)
        return index.astype(np.int64), self.labels[index]

    def calibrate_beta(self, clean_waves: np.ndarray) -> Dict[str, float]:
        """Set beta so a typical correct lead is BETA_TARGET_DECADES sharp."""
        scores = self.alignments(clean_waves)
        if scores.size == 0 or scores.shape[1] < 2:
            self.beta = 1.0
            self.beta_source = "default"
            return {
                "beta": 1.0,
                "beta_median_gap": 0.0,
                "beta_source": 0.0,
                "beta_star_refused": self.BETA_STAR_REFERENCE,
            }
        ordered = np.sort(scores, axis=1)
        gaps = ordered[:, -1] - ordered[:, -2]
        median_gap = float(np.median(gaps))
        beta = self.BETA_TARGET_DECADES / max(median_gap, 1e-3)
        self.beta = float(beta)
        self.beta_source = "calibrated"
        return {
            "beta": float(beta),
            "beta_median_gap": median_gap,
            "beta_source": 1.0,
            "beta_star_refused": self.BETA_STAR_REFERENCE,
        }

    def accuracy(self, waves: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Snap accuracy and mean top-1 alignment of the bank."""
        scores = self.alignments(waves)
        if scores.size == 0:
            return {"prototype_accuracy": float("nan"), "prototype_alignment": float("nan")}
        _, predicted = self.snap(waves)
        return {
            "prototype_accuracy": float(np.mean(predicted == np.asarray(labels))),
            "prototype_alignment": float(np.mean(scores.max(axis=1))),
        }

    # -------------------------------------------------------------- zone c io
    def write_to_memory(
        self,
        memory: "ZoneCMemory",
        tier: int = TIER_AXIOMATIC,
        offset: int = 0,
    ) -> int:
        """Persist every prototype as an axiomatic Zone C engram."""
        written = 0
        for wave, label in zip(self.phi, self.labels):
            memory.write(wave, label=int(label) + int(offset), tier=int(tier))
            written += 1
        return written

    def load_from_memory(
        self,
        memory: "ZoneCMemory",
        offset: int = 0,
        block: Optional[Tuple[int, int]] = None,
    ) -> int:
        """Reload the bank from the axiomatic engrams already in Zone C.

        Decision D1: block restricts the reload to one task's global label
        range, and offset converts a global label back to its local class id.
        """
        rows = memory.axiomatic_engrams()
        if block is not None:
            rows = [row for row in rows if int(block[0]) <= int(row[1]) < int(block[1])]
        if not rows:
            self.phi = np.zeros((0, self.dim), dtype=np.complex64)
            self.labels = np.zeros(0, dtype=np.int64)
            self.counts = np.zeros(0, dtype=np.int64)
            return 0
        waves = memory._fetch([int(row[0]) for row in rows])
        kept = [
            (wave, int(label) - int(offset))
            for wave, (_, label) in zip(waves, rows)
            if int(wave.size) == self.dim
        ]
        if not kept:
            return 0
        self.phi = np.stack([_unit(w.astype(np.complex64)) for w, _ in kept])
        self.labels = np.asarray([int(label) for _, label in kept], dtype=np.int64)
        self.counts = np.ones(len(kept), dtype=np.int64)
        return int(len(kept))

    def report(self) -> Dict[str, float]:
        return {
            "prototypes": float(len(self)),
            "prototype_beta": float(self.beta),
            "prototype_beta_calibrated": 1.0 if self.beta_source == "calibrated" else 0.0,
            "prototype_beta_star_refused": self.BETA_STAR_REFERENCE,
        }
