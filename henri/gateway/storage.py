"""Persistent gateway storage for TimescaleDB (PostgreSQL) and SQLite.

Evidence label: OBSERVED for every row written, read, or verified.

Manages four additive tables alongside Zone C's `holarch_engrams` and
`engram_meta`:
  1. `gateway_api_keys`     - Scoped API-key hashes and per-tenant policies.
  2. `gateway_cartridges`   - Persistent DigitalTwin D=65536 rank-r cartridges
                              (stores compressed [rows, dim] complex arrays;
                               strictly zero dense D x D allocations — C7).
  3. `gateway_receipts`     - Per-namespace Ed25519-signed hash-chained ledger.
  4. `gateway_spend_ledger` - Atomic per-tenant SpendGuard budget accounting.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.cartridge_bank import Cartridge, CartridgeBank
from henri.gateway.config import ensure_parent_dir
from henri.gateway.signing import GENESIS


@dataclass
class StoredApiKey:
    key_id: str
    secret_hash: str
    tenant_id: str
    allowed_namespaces: Tuple[str, ...]
    default_namespace: str
    rpm_limit: int = 120
    burst_limit: int = 30
    llm_call_budget: int = 50
    allow_cascade: bool = True
    allow_sandbox_exec: bool = True
    revoked: bool = False


@dataclass
class StoredCartridgeRow:
    tenant_id: str
    cartridge_index: int
    subject: str
    dim: int
    rank: int
    rows: int
    dtype: str
    keys_blob: bytes
    values_blob: bytes
    sha256: str


def _pack_array(arr: np.ndarray) -> bytes:
    arr_c = np.ascontiguousarray(arr)
    return zlib.compress(arr_c.tobytes(), level=6)


def _unpack_array(blob: bytes, rows: int, dim: int, dtype_str: str) -> np.ndarray:
    raw = zlib.decompress(blob)
    dt = np.dtype(dtype_str)
    arr = np.frombuffer(raw, dtype=dt)
    expected = int(rows) * int(dim)
    if arr.size != expected:
        raise ValueError(
            f"corrupted cartridge blob: got {arr.size} elements, expected {expected}"
        )
    return arr.reshape((int(rows), int(dim))).copy()


def _cartridge_digest(
    tenant_id: str,
    idx: int,
    dim: int,
    rank: int,
    rows: int,
    dtype_str: str,
    keys_blob: bytes,
    values_blob: bytes,
) -> str:
    h = hashlib.sha256()
    h.update(f"{tenant_id}|{idx}|{dim}|{rank}|{rows}|{dtype_str}|".encode("utf-8"))
    h.update(keys_blob)
    h.update(b"|")
    h.update(values_blob)
    return h.hexdigest()


class GatewayStore:
    """Dual-backend (TimescaleDB / SQLite) persistence for the unified gateway."""

    def __init__(
        self,
        *,
        backend: str = "sqlite",
        sqlite_path: str = "data/gateway/henri_gateway.sqlite3",
        timescale_dsn: str = "postgresql://postgres:henri_local_only@localhost:55432/henri",
        connect_timeout: int = 5,
    ) -> None:
        self._lock = threading.RLock()
        self.requested_backend = str(backend).lower()
        self.backend_reason = ""
        self._pg_conn = None
        self._sq_conn: Optional[sqlite3.Connection] = None

        if self.requested_backend in {"timescale", "timescaledb"}:
            self._init_postgres(timescale_dsn, connect_timeout)
            self.backend = "timescaledb"
            self.backend_reason = "explicit timescaledb"
        elif self.requested_backend == "auto":
            try:
                self._init_postgres(timescale_dsn, connect_timeout)
                self.backend = "timescaledb"
                self.backend_reason = "auto connected to timescaledb"
            except Exception as exc:
                self._init_sqlite(sqlite_path)
                self.backend = "sqlite"
                self.backend_reason = f"auto fallback to sqlite: {exc}"
        elif self.requested_backend == "sqlite":
            self._init_sqlite(sqlite_path)
            self.backend = "sqlite"
            self.backend_reason = "explicit sqlite"
        else:
            raise ValueError(f"unsupported backend: {backend!r}")

    def _init_sqlite(self, sqlite_path: str) -> None:
        if sqlite_path != ":memory:":
            ensure_parent_dir(sqlite_path)
        self._sq_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._sq_conn.execute("PRAGMA journal_mode=WAL")
        self._sq_conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate_sqlite()

    def _init_postgres(self, dsn: str, connect_timeout: int) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("BLOCKED: psycopg is not installed") from exc
        self._pg_conn = psycopg.connect(
            dsn, autocommit=True, connect_timeout=connect_timeout
        )
        self._migrate_postgres()

    # ------------------------------------------------------------ migrations
    def _migrate_sqlite(self) -> None:
        assert self._sq_conn is not None
        with self._lock:
            cur = self._sq_conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS gateway_api_keys (
                    key_id              TEXT PRIMARY KEY,
                    secret_hash         TEXT NOT NULL,
                    tenant_id           TEXT NOT NULL,
                    allowed_namespaces  TEXT NOT NULL,
                    default_namespace   TEXT NOT NULL,
                    rpm_limit           INTEGER NOT NULL DEFAULT 120,
                    burst_limit         INTEGER NOT NULL DEFAULT 30,
                    llm_call_budget     INTEGER NOT NULL DEFAULT 50,
                    allow_cascade       INTEGER NOT NULL DEFAULT 1,
                    allow_sandbox_exec  INTEGER NOT NULL DEFAULT 1,
                    revoked             INTEGER NOT NULL DEFAULT 0,
                    created_at          TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gateway_cartridges (
                    tenant_id           TEXT NOT NULL,
                    cartridge_index     INTEGER NOT NULL,
                    subject             TEXT NOT NULL DEFAULT '',
                    dim                 INTEGER NOT NULL,
                    rank                INTEGER NOT NULL,
                    rows                INTEGER NOT NULL,
                    dtype               TEXT NOT NULL,
                    keys_blob           BLOB NOT NULL,
                    values_blob         BLOB NOT NULL,
                    sha256              TEXT NOT NULL,
                    created_at          TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, cartridge_index)
                );
                CREATE TABLE IF NOT EXISTS gateway_receipts (
                    namespace           TEXT NOT NULL,
                    seq                 INTEGER NOT NULL,
                    created_at          TEXT NOT NULL,
                    endpoint            TEXT NOT NULL,
                    parent_sha256       TEXT NOT NULL,
                    payload_sha256      TEXT NOT NULL,
                    key_id              TEXT NOT NULL,
                    signature_b64u      TEXT NOT NULL,
                    evidence_label      TEXT NOT NULL,
                    provenance          TEXT NOT NULL,
                    body_json           TEXT NOT NULL,
                    PRIMARY KEY (namespace, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_gw_receipts_ns_seq
                    ON gateway_receipts(namespace, seq);
                CREATE TABLE IF NOT EXISTS gateway_spend_ledger (
                    tenant_id           TEXT NOT NULL,
                    window_key          TEXT NOT NULL,
                    calls_used          INTEGER NOT NULL DEFAULT 0,
                    tokens_prompt       INTEGER NOT NULL DEFAULT 0,
                    tokens_completion   INTEGER NOT NULL DEFAULT 0,
                    updated_at          TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, window_key)
                );
                CREATE TABLE IF NOT EXISTS gateway_tenant_state (
                    namespace           TEXT NOT NULL,
                    kind                TEXT NOT NULL,
                    state_key           TEXT NOT NULL,
                    payload_json        TEXT NOT NULL,
                    updated_at          TEXT NOT NULL,
                    PRIMARY KEY (namespace, kind, state_key)
                );
                """
            )
            self._sq_conn.commit()

    def _migrate_postgres(self) -> None:
        assert self._pg_conn is not None
        with self._lock:
            with self._pg_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gateway_api_keys (
                        key_id              TEXT PRIMARY KEY,
                        secret_hash         TEXT NOT NULL,
                        tenant_id           TEXT NOT NULL,
                        allowed_namespaces  TEXT[] NOT NULL,
                        default_namespace   TEXT NOT NULL,
                        rpm_limit           INTEGER NOT NULL DEFAULT 120,
                        burst_limit         INTEGER NOT NULL DEFAULT 30,
                        llm_call_budget     INTEGER NOT NULL DEFAULT 50,
                        allow_cascade       BOOLEAN NOT NULL DEFAULT TRUE,
                        allow_sandbox_exec  BOOLEAN NOT NULL DEFAULT TRUE,
                        revoked             BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gateway_cartridges (
                        tenant_id           TEXT NOT NULL,
                        cartridge_index     INTEGER NOT NULL,
                        subject             TEXT NOT NULL DEFAULT '',
                        dim                 INTEGER NOT NULL,
                        rank                INTEGER NOT NULL,
                        rows                INTEGER NOT NULL,
                        dtype               TEXT NOT NULL,
                        keys_blob           BYTEA NOT NULL,
                        values_blob         BYTEA NOT NULL,
                        sha256              TEXT NOT NULL,
                        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (tenant_id, cartridge_index)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gateway_receipts (
                        namespace           TEXT NOT NULL,
                        seq                 BIGINT NOT NULL,
                        created_at          TIMESTAMPTZ NOT NULL,
                        endpoint            TEXT NOT NULL,
                        parent_sha256       TEXT NOT NULL,
                        payload_sha256      TEXT NOT NULL,
                        key_id              TEXT NOT NULL,
                        signature_b64u      TEXT NOT NULL,
                        evidence_label      TEXT NOT NULL,
                        provenance          TEXT NOT NULL,
                        body_json           JSONB NOT NULL,
                        PRIMARY KEY (namespace, seq, created_at)
                    )
                    """
                )
                cur.execute(
                    "SELECT create_hypertable('gateway_receipts', 'created_at', "
                    "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE)"
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gateway_spend_ledger (
                        tenant_id           TEXT NOT NULL,
                        window_key          TEXT NOT NULL,
                        calls_used          INTEGER NOT NULL DEFAULT 0,
                        tokens_prompt       BIGINT NOT NULL DEFAULT 0,
                        tokens_completion   BIGINT NOT NULL DEFAULT 0,
                        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (tenant_id, window_key)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gateway_tenant_state (
                        namespace           TEXT NOT NULL,
                        kind                TEXT NOT NULL,
                        state_key           TEXT NOT NULL,
                        payload_json        TEXT NOT NULL,
                        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (namespace, kind, state_key)
                    )
                    """
                )

    def ping(self) -> Tuple[bool, str]:
        try:
            with self._lock:
                if self.backend == "timescaledb":
                    assert self._pg_conn is not None
                    with self._pg_conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
                else:
                    assert self._sq_conn is not None
                    self._sq_conn.execute("SELECT 1").fetchone()
            return True, self.backend
        except Exception as exc:
            return False, str(exc)

    def close(self) -> None:
        with self._lock:
            if self._sq_conn is not None:
                self._sq_conn.close()
                self._sq_conn = None
            if self._pg_conn is not None:
                self._pg_conn.close()
                self._pg_conn = None

    # -------------------------------------------------------------- API keys
    def upsert_api_key(self, record: StoredApiKey) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gateway_api_keys (
                            key_id, secret_hash, tenant_id, allowed_namespaces,
                            default_namespace, rpm_limit, burst_limit, llm_call_budget,
                            allow_cascade, allow_sandbox_exec, revoked, created_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (key_id) DO UPDATE SET
                            secret_hash = EXCLUDED.secret_hash,
                            tenant_id = EXCLUDED.tenant_id,
                            allowed_namespaces = EXCLUDED.allowed_namespaces,
                            default_namespace = EXCLUDED.default_namespace,
                            rpm_limit = EXCLUDED.rpm_limit,
                            burst_limit = EXCLUDED.burst_limit,
                            llm_call_budget = EXCLUDED.llm_call_budget,
                            allow_cascade = EXCLUDED.allow_cascade,
                            allow_sandbox_exec = EXCLUDED.allow_sandbox_exec,
                            revoked = EXCLUDED.revoked
                        """,
                        (
                            record.key_id,
                            record.secret_hash,
                            record.tenant_id,
                            list(record.allowed_namespaces),
                            record.default_namespace,
                            int(record.rpm_limit),
                            int(record.burst_limit),
                            int(record.llm_call_budget),
                            bool(record.allow_cascade),
                            bool(record.allow_sandbox_exec),
                            bool(record.revoked),
                            now,
                        ),
                    )
            else:
                assert self._sq_conn is not None
                self._sq_conn.execute(
                    """
                    INSERT INTO gateway_api_keys (
                        key_id, secret_hash, tenant_id, allowed_namespaces,
                        default_namespace, rpm_limit, burst_limit, llm_call_budget,
                        allow_cascade, allow_sandbox_exec, revoked, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key_id) DO UPDATE SET
                        secret_hash = excluded.secret_hash,
                        tenant_id = excluded.tenant_id,
                        allowed_namespaces = excluded.allowed_namespaces,
                        default_namespace = excluded.default_namespace,
                        rpm_limit = excluded.rpm_limit,
                        burst_limit = excluded.burst_limit,
                        llm_call_budget = excluded.llm_call_budget,
                        allow_cascade = excluded.allow_cascade,
                        allow_sandbox_exec = excluded.allow_sandbox_exec,
                        revoked = excluded.revoked
                    """,
                    (
                        record.key_id,
                        record.secret_hash,
                        record.tenant_id,
                        json.dumps(list(record.allowed_namespaces)),
                        record.default_namespace,
                        int(record.rpm_limit),
                        int(record.burst_limit),
                        int(record.llm_call_budget),
                        int(bool(record.allow_cascade)),
                        int(bool(record.allow_sandbox_exec)),
                        int(bool(record.revoked)),
                        now,
                    ),
                )
                self._sq_conn.commit()

    def get_api_key(self, key_id: str) -> Optional[StoredApiKey]:
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT key_id, secret_hash, tenant_id, allowed_namespaces,
                               default_namespace, rpm_limit, burst_limit, llm_call_budget,
                               allow_cascade, allow_sandbox_exec, revoked
                        FROM gateway_api_keys WHERE key_id = %s
                        """,
                        (key_id,),
                    )
                    row = cur.fetchone()
                if row is None:
                    return None
                return StoredApiKey(
                    key_id=str(row[0]),
                    secret_hash=str(row[1]),
                    tenant_id=str(row[2]),
                    allowed_namespaces=tuple(str(x) for x in row[3]),
                    default_namespace=str(row[4]),
                    rpm_limit=int(row[5]),
                    burst_limit=int(row[6]),
                    llm_call_budget=int(row[7]),
                    allow_cascade=bool(row[8]),
                    allow_sandbox_exec=bool(row[9]),
                    revoked=bool(row[10]),
                )
            else:
                assert self._sq_conn is not None
                row = self._sq_conn.execute(
                    """
                    SELECT key_id, secret_hash, tenant_id, allowed_namespaces,
                           default_namespace, rpm_limit, burst_limit, llm_call_budget,
                           allow_cascade, allow_sandbox_exec, revoked
                    FROM gateway_api_keys WHERE key_id = ?
                    """,
                    (key_id,),
                ).fetchone()
                if row is None:
                    return None
                return StoredApiKey(
                    key_id=str(row[0]),
                    secret_hash=str(row[1]),
                    tenant_id=str(row[2]),
                    allowed_namespaces=tuple(str(x) for x in json.loads(row[3])),
                    default_namespace=str(row[4]),
                    rpm_limit=int(row[5]),
                    burst_limit=int(row[6]),
                    llm_call_budget=int(row[7]),
                    allow_cascade=bool(row[8]),
                    allow_sandbox_exec=bool(row[9]),
                    revoked=bool(row[10]),
                )

    def count_api_keys(self) -> int:
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM gateway_api_keys WHERE NOT revoked")
                    return int(cur.fetchone()[0])
            assert self._sq_conn is not None
            row = self._sq_conn.execute(
                "SELECT COUNT(*) FROM gateway_api_keys WHERE revoked = 0"
            ).fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------ cartridges
    def save_cartridge(self, tenant_id: str, cartridge: Cartridge) -> str:
        """Persist one rank-r cartridge. Zero dense D x D allocation (C7)."""
        dtype_str = str(np.dtype(cartridge.dtype))
        keys_blob = _pack_array(cartridge.keys_raw)
        values_blob = _pack_array(cartridge.values)
        digest = _cartridge_digest(
            tenant_id,
            int(cartridge.index),
            int(cartridge.dim),
            int(cartridge.rank),
            int(cartridge.rows),
            dtype_str,
            keys_blob,
            values_blob,
        )
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gateway_cartridges (
                            tenant_id, cartridge_index, subject, dim, rank, rows,
                            dtype, keys_blob, values_blob, sha256, created_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (tenant_id, cartridge_index) DO UPDATE SET
                            subject = EXCLUDED.subject,
                            dim = EXCLUDED.dim,
                            rank = EXCLUDED.rank,
                            rows = EXCLUDED.rows,
                            dtype = EXCLUDED.dtype,
                            keys_blob = EXCLUDED.keys_blob,
                            values_blob = EXCLUDED.values_blob,
                            sha256 = EXCLUDED.sha256
                        """,
                        (
                            tenant_id,
                            int(cartridge.index),
                            str(cartridge.subject),
                            int(cartridge.dim),
                            int(cartridge.rank),
                            int(cartridge.rows),
                            dtype_str,
                            keys_blob,
                            values_blob,
                            digest,
                            now,
                        ),
                    )
            else:
                assert self._sq_conn is not None
                self._sq_conn.execute(
                    """
                    INSERT INTO gateway_cartridges (
                        tenant_id, cartridge_index, subject, dim, rank, rows,
                        dtype, keys_blob, values_blob, sha256, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(tenant_id, cartridge_index) DO UPDATE SET
                        subject = excluded.subject,
                        dim = excluded.dim,
                        rank = excluded.rank,
                        rows = excluded.rows,
                        dtype = excluded.dtype,
                        keys_blob = excluded.keys_blob,
                        values_blob = excluded.values_blob,
                        sha256 = excluded.sha256
                    """,
                    (
                        tenant_id,
                        int(cartridge.index),
                        str(cartridge.subject),
                        int(cartridge.dim),
                        int(cartridge.rank),
                        int(cartridge.rows),
                        dtype_str,
                        keys_blob,
                        values_blob,
                        digest,
                        now,
                    ),
                )
                self._sq_conn.commit()
        return digest

    def load_cartridge_bank(self, tenant_id: str) -> Optional[CartridgeBank]:
        """Hydrate a tenant's CartridgeBank from persistent storage, verifying digests."""
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT cartridge_index, subject, dim, rank, rows,
                               dtype, keys_blob, values_blob, sha256
                        FROM gateway_cartridges
                        WHERE tenant_id = %s
                        ORDER BY cartridge_index ASC
                        """,
                        (tenant_id,),
                    )
                    rows = cur.fetchall()
            else:
                assert self._sq_conn is not None
                rows = self._sq_conn.execute(
                    """
                    SELECT cartridge_index, subject, dim, rank, rows,
                           dtype, keys_blob, values_blob, sha256
                    FROM gateway_cartridges
                    WHERE tenant_id = ?
                    ORDER BY cartridge_index ASC
                    """,
                    (tenant_id,),
                ).fetchall()

        if not rows:
            return None

        first_dim = int(rows[0][2])
        first_rank = int(rows[0][3])
        first_dtype = np.dtype(str(rows[0][5]))
        bank = CartridgeBank(dim=first_dim, rank=first_rank, dtype=first_dtype)
        for row in rows:
            idx, subject, dim, rank, n_rows, dtype_str, k_blob, v_blob, expected_sha = row
            k_bytes = bytes(k_blob)
            v_bytes = bytes(v_blob)
            actual = _cartridge_digest(
                tenant_id, int(idx), int(dim), int(rank), int(n_rows),
                str(dtype_str), k_bytes, v_bytes,
            )
            if actual != str(expected_sha):
                raise RuntimeError(
                    f"cartridge integrity failure for tenant={tenant_id!r} index={idx}"
                )
            keys = _unpack_array(k_bytes, int(n_rows), int(dim), str(dtype_str))
            vals = _unpack_array(v_bytes, int(n_rows), int(dim), str(dtype_str))
            bank.add_cartridge(keys, vals, subject=str(subject))
        bank.finalize()
        return bank

    # -------------------------------------------------------------- receipts
    def get_receipt_head(self, namespace: str) -> Tuple[int, str]:
        """Return (next_seq, parent_sha256) for `namespace`."""
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT seq, payload_sha256
                        FROM gateway_receipts
                        WHERE namespace = %s
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        (namespace,),
                    )
                    row = cur.fetchone()
            else:
                assert self._sq_conn is not None
                row = self._sq_conn.execute(
                    """
                    SELECT seq, payload_sha256
                    FROM gateway_receipts
                    WHERE namespace = ?
                    ORDER BY seq DESC
                    LIMIT 1
                    """,
                    (namespace,),
                ).fetchone()
        if row is None:
            return 1, GENESIS
        return int(row[0]) + 1, str(row[1])

    def append_receipt(self, receipt: Dict[str, Any]) -> None:
        sig = receipt.get("signature") or {}
        ns = str(receipt["namespace"])
        seq = int(receipt["seq"])
        created_at = str(receipt["created_utc"])
        endpoint = str(receipt["endpoint"])
        parent_sha = str(receipt["parent"])
        payload_sha = str(receipt["sha256"])
        kid = str(sig.get("kid", ""))
        sig_b64u = str(sig.get("sig", ""))
        evidence = str(receipt.get("evidence_label", "OBSERVED"))
        provenance = str(receipt.get("provenance", "wave_only"))
        body_str = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gateway_receipts (
                            namespace, seq, created_at, endpoint, parent_sha256,
                            payload_sha256, key_id, signature_b64u, evidence_label,
                            provenance, body_json
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        """,
                        (
                            ns, seq, created_at, endpoint, parent_sha, payload_sha,
                            kid, sig_b64u, evidence, provenance, body_str,
                        ),
                    )
            else:
                assert self._sq_conn is not None
                self._sq_conn.execute(
                    """
                    INSERT INTO gateway_receipts (
                        namespace, seq, created_at, endpoint, parent_sha256,
                        payload_sha256, key_id, signature_b64u, evidence_label,
                        provenance, body_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        ns, seq, created_at, endpoint, parent_sha, payload_sha,
                        kid, sig_b64u, evidence, provenance, body_str,
                    ),
                )
                self._sq_conn.commit()

    def list_receipts(self, namespace: str, limit: int = 16) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT body_json
                        FROM gateway_receipts
                        WHERE namespace = %s
                        ORDER BY seq DESC
                        LIMIT %s
                        """,
                        (namespace, lim),
                    )
                    rows = cur.fetchall()
                items = [
                    r[0] if isinstance(r[0], dict) else json.loads(r[0])
                    for r in reversed(rows)
                ]
                return items
            assert self._sq_conn is not None
            rows = self._sq_conn.execute(
                """
                SELECT body_json
                FROM gateway_receipts
                WHERE namespace = ?
                ORDER BY seq DESC
                LIMIT ?
                """,
                (namespace, lim),
            ).fetchall()
            return [json.loads(r[0]) for r in reversed(rows)]

    def count_receipts(self, namespace: str) -> int:
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM gateway_receipts WHERE namespace = %s",
                        (namespace,),
                    )
                    return int(cur.fetchone()[0])
            assert self._sq_conn is not None
            row = self._sq_conn.execute(
                "SELECT COUNT(*) FROM gateway_receipts WHERE namespace = ?",
                (namespace,),
            ).fetchone()
            return int(row[0]) if row else 0

    def verify_receipt_chain(self, namespace: str) -> Tuple[bool, int, str]:
        """Verify contiguous parent->sha256 chain for a namespace."""
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT seq, parent_sha256, payload_sha256
                        FROM gateway_receipts
                        WHERE namespace = %s
                        ORDER BY seq ASC
                        """,
                        (namespace,),
                    )
                    rows = cur.fetchall()
            else:
                assert self._sq_conn is not None
                rows = self._sq_conn.execute(
                    """
                    SELECT seq, parent_sha256, payload_sha256
                    FROM gateway_receipts
                    WHERE namespace = ?
                    ORDER BY seq ASC
                    """,
                    (namespace,),
                ).fetchall()
        expected_parent = GENESIS
        expected_seq = 1
        for seq, parent_sha, payload_sha in rows:
            if int(seq) != expected_seq or str(parent_sha) != expected_parent:
                return False, len(rows), expected_parent
            expected_parent = str(payload_sha)
            expected_seq += 1
        return True, len(rows), expected_parent

    # ---------------------------------------------------------- spend ledger
    def reserve_llm_call(
        self,
        tenant_id: str,
        budget_limit: int,
        *,
        window_key: Optional[str] = None,
    ) -> Tuple[bool, int, int]:
        """Atomically reserve 1 Arm4 LLM escalation call.

        Returns `(granted, calls_used, budget_limit)`.
        """
        win = window_key or time.strftime("%Y-%m-%d", time.gmtime())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        lim = max(0, int(budget_limit))
        if lim == 0:
            return False, 0, 0
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gateway_spend_ledger
                            (tenant_id, window_key, calls_used, updated_at)
                        VALUES (%s, %s, 1, %s)
                        ON CONFLICT (tenant_id, window_key) DO UPDATE
                            SET calls_used = gateway_spend_ledger.calls_used + 1,
                                updated_at = EXCLUDED.updated_at
                            WHERE gateway_spend_ledger.calls_used < %s
                        RETURNING calls_used
                        """,
                        (tenant_id, win, now, lim),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        return True, int(row[0]), lim
                    cur.execute(
                        "SELECT calls_used FROM gateway_spend_ledger "
                        "WHERE tenant_id = %s AND window_key = %s",
                        (tenant_id, win),
                    )
                    cur_row = cur.fetchone()
                    return False, int(cur_row[0]) if cur_row else lim, lim
            else:
                assert self._sq_conn is not None
                cur = self._sq_conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                try:
                    row = cur.execute(
                        "SELECT calls_used FROM gateway_spend_ledger "
                        "WHERE tenant_id = ? AND window_key = ?",
                        (tenant_id, win),
                    ).fetchone()
                    used = int(row[0]) if row else 0
                    if used >= lim:
                        self._sq_conn.rollback()
                        return False, used, lim
                    new_used = used + 1
                    cur.execute(
                        """
                        INSERT INTO gateway_spend_ledger
                            (tenant_id, window_key, calls_used, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(tenant_id, window_key) DO UPDATE SET
                            calls_used = excluded.calls_used,
                            updated_at = excluded.updated_at
                        """,
                        (tenant_id, win, new_used, now),
                    )
                    self._sq_conn.commit()
                    return True, new_used, lim
                except Exception:
                    self._sq_conn.rollback()
                    raise

    def record_llm_tokens(
        self,
        tenant_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        window_key: Optional[str] = None,
    ) -> None:
        win = window_key or time.strftime("%Y-%m-%d", time.gmtime())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE gateway_spend_ledger
                        SET tokens_prompt = tokens_prompt + %s,
                            tokens_completion = tokens_completion + %s,
                            updated_at = %s
                        WHERE tenant_id = %s AND window_key = %s
                        """,
                        (int(prompt_tokens), int(completion_tokens), now, tenant_id, win),
                    )
            else:
                assert self._sq_conn is not None
                self._sq_conn.execute(
                    """
                    UPDATE gateway_spend_ledger
                    SET tokens_prompt = tokens_prompt + ?,
                        tokens_completion = tokens_completion + ?,
                        updated_at = ?
                    WHERE tenant_id = ? AND window_key = ?
                    """,
                    (int(prompt_tokens), int(completion_tokens), now, tenant_id, win),
                )
                self._sq_conn.commit()

    # ---------------------------------------------------------- tenant state
    def put_tenant_state(
        self, namespace: str, kind: str, state_key: str, payload: Dict[str, Any]
    ) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        blob = json.dumps(payload, sort_keys=True, default=str)
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gateway_tenant_state
                            (namespace, kind, state_key, payload_json, updated_at)
                        VALUES (%s,%s,%s,%s,%s)
                        ON CONFLICT (namespace, kind, state_key) DO UPDATE SET
                            payload_json = EXCLUDED.payload_json,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (namespace, kind, state_key, blob, now),
                    )
            else:
                assert self._sq_conn is not None
                self._sq_conn.execute(
                    """
                    INSERT INTO gateway_tenant_state
                        (namespace, kind, state_key, payload_json, updated_at)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(namespace, kind, state_key) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (namespace, kind, state_key, blob, now),
                )
                self._sq_conn.commit()

    def get_tenant_state(
        self, namespace: str, kind: str, state_key: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self.backend == "timescaledb":
                assert self._pg_conn is not None
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT payload_json FROM gateway_tenant_state
                        WHERE namespace = %s AND kind = %s AND state_key = %s
                        """,
                        (namespace, kind, state_key),
                    )
                    row = cur.fetchone()
            else:
                assert self._sq_conn is not None
                row = self._sq_conn.execute(
                    """
                    SELECT payload_json FROM gateway_tenant_state
                    WHERE namespace = ? AND kind = ? AND state_key = ?
                    """,
                    (namespace, kind, state_key),
                ).fetchone()
        return json.loads(row[0]) if row else None
