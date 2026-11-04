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
                weight REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_engram_dim ON engram(dim, engram_id)"
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
        )

    def append(self, engram: Engram) -> None:
        self.append_many([engram])

    def append_many(self, engrams: Sequence[Engram]) -> None:
        if not engrams:
            return
        dim = int(np.asarray(engrams[0].wave).size)
        rows = [self._row(e, dim) for e in engrams]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO engram VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
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
                f"gram, lyapunov, info_gain, tier, weight FROM engram "
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
                )
            )
        by_id = {e.engram_id: e for e in out}
        return [by_id[i] for i in ids if i in by_id]

    def metadata(self, dim: int) -> List[Tuple[int, float, int, bytes, int, float]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT engram_id, recorded_at, label, gram, tier, weight "
                "FROM engram WHERE dim = ? ORDER BY engram_id",
                (int(dim),),
            ).fetchall()
        return [(int(r[0]), float(r[1]), int(r[2]), bytes(r[3]), int(r[4]), float(r[5])) for r in rows]

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
                    PRIMARY KEY (engram_id, recorded_at)
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
        )

    def append(self, engram: Engram) -> None:
        self.append_many([engram])

    def append_many(self, engrams: Sequence[Engram]) -> None:
        if not engrams:
            return
        dim = int(np.asarray(engrams[0].wave).size)
        with self._lock, self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} VALUES "
                "(%s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
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
                f"gram, lyapunov, info_gain, tier, weight FROM {self.table} "
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
                )
            )
        by_id = {e.engram_id: e for e in out}
        return [by_id[i] for i in ids if i in by_id]

    def metadata(self, dim: int) -> List[Tuple[int, float, int, bytes, int, float]]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"SELECT engram_id, EXTRACT(EPOCH FROM recorded_at), label, gram, tier, weight "
                f"FROM {self.table} WHERE dim = %s ORDER BY engram_id",
                (int(dim),),
            )
            rows = cur.fetchall()
        return [
            (int(r[0]), float(r[1]), int(r[2]), bytes(r[3]), int(r[4]), float(r[5]))
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
        self._ensure(64)
        self._load()

    # ------------------------------------------------------------- index io
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
        for engram_id, ts, label, gram_blob, tier, weight in rows:
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
            self._tier[index] = int(tier)
            self._weight[index] = float(weight)
            self._size += 1
        self._next_id = highest + 1
        for engram_id in self._ids[: self._size]:
            self._prefetched.add(int(engram_id))

    # ---------------------------------------------------------------- write
    def write(
        self,
        wave: np.ndarray,
        label: int = -1,
        info_gain: float = 0.0,
        tier: int = 1,
        ts: Optional[float] = None,
    ) -> Engram:
        """Persist one engram. Returns the stored record."""
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
            )
            self.store.append(record)
            self._ids[index] = engram_id
            self._ts[index] = stamp
            self._labels[index] = int(label)
            self._grams[index] = gram
            self._tier[index] = int(tier)
            self._weight[index] = 1.0
            self._size += 1
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
        resolved: Dict[int, np.ndarray] = {}
        with self._lock:
            for engram_id in ids:
                if engram_id in self._cache:
                    resolved[int(engram_id)] = self._cache[int(engram_id)]
            missing = [int(i) for i in ids if int(i) not in resolved]
            if missing:
                for record in self.store.fetch(missing):
                    resolved[record.engram_id] = record.wave
                    self._put_cache(record.engram_id, record.wave)
                self._cache_misses += len(missing)
            if len(missing) < len(ids):
                self._cache_hits += len(ids) - len(missing)
            return [resolved[int(i)] for i in ids if int(i) in resolved]

    def query(
        self,
        wave: np.ndarray,
        k: int = 8,
        avoid_ids: Sequence[int] = (),
        candidate_multiplier: int = 6,
        task: Optional[str] = None,
    ) -> MemoryRead:
        """Retrieve the k best engrams and fuse them into one context wave.

        Decision D1: when task is given, only engrams whose stored label lies
        in that task's disjoint label block are candidates. A task-scoped
        query with no candidate in its block returns an empty read, which the
        C1 rule turns into a veto for inference.
        """
        start = time.perf_counter()
        query_wave = _unit(wave)
        query_gram = (self.projection @ query_wave).astype(np.complex64)
        with self._lock:
            size = self._size
            if size == 0:
                self._last_query = query_wave
                return MemoryRead([], np.zeros(self.dim, dtype=np.complex64), 0.0,
                                  (time.perf_counter() - start) * 1e3, 0, True, 0.0, None)
            grams = self._grams[:size]
            ids = self._ids[:size]
            labels = self._labels[:size]
            norms = np.linalg.norm(grams, axis=1)
            scores = np.real(grams @ np.conj(query_gram)) / np.maximum(
                norms * (float(np.linalg.norm(query_gram)) + _EPS), _EPS
            )
            scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
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
                int(ids[i]): (int(self._labels[i]), float(self._ts[i]), int(self._tier[i]))
                for i in order
            }
        waves = self._fetch(candidate_ids)
        by_id = dict(zip(candidate_ids, waves))
        exact = []
        for engram_id in candidate_ids:
            if engram_id not in by_id:
                continue
            exact.append((engram_id, coherence(query_wave, by_id[engram_id])))
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
    def context_tensor(
        self, wave: np.ndarray, k: int = 8, task: Optional[str] = None
    ) -> Tuple[np.ndarray, MemoryRead]:
        """Fixed (context_tokens, context_dim) tensor at every store size."""
        read = self.query(wave, k=k, task=task)
        tokens = int(self.cfg.context_tokens)
        if read.hits:
            chunks = read.context.reshape(tokens, self.cfg.chunk)
            prefix = (
                chunks.real @ self.context_proj + chunks.imag @ self.context_proj
            ) / np.sqrt(2.0)
        else:
            prefix = np.zeros((tokens, int(self.cfg.context_dim)), dtype=np.float32)
        return prefix.astype(np.float32), read

    def axiomatic_engrams(self) -> List[Tuple[int, int]]:
        """(engram_id, label) for every axiomatic (tier-3) row.

        Tier-3 rows are the clean prototype bank of Decision 2. They are
        never pruned, so the attractor bank survives compaction and decay.
        """
        rows = self.store.metadata(self.dim)
        return [(int(row[0]), int(row[2])) for row in rows if int(row[4]) >= TIER_AXIOMATIC]

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
