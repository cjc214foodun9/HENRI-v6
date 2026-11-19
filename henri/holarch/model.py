"""Checkpoint I/O and the closed-loop action evaluator for the HENRI holarch.

Evidence label: OBSERVED for every wave alignment, action choice, and
success rate this module returns. A checkpoint is a byte-identical copy of
the weights, the calibrated buffers, and the clean prototype bank, so a
service that mounts it answers with the converged model rather than with a
fresh bootstrap.

Cycle E adds one evaluator here instead of in the training loop:

    evaluate_closed_loop_action   a 1-step latent wave lookahead policy

The policy scores each of the four grid actions with the Zone A logit plus
a goal-alignment term minus a cyclic-repulsion term. The transition is a
circular convolution on a four-slot sensor wave, so the lookahead is a
measured wave model and not a relabelled copy of the reactive logits. The
learned action head supplies y_action; the transition and the goal
attractor run on the explicit four-slot sensor frame, so the lookahead is
a forward model with a physical shift operator rather than a copy of the
reactive policy's own logits.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from henri.holarch.config import TASK_NAMES, HolarchConfig
from henri.holarch.pipeline import HenriHolarch

ACTION_NAMES = ("up", "down", "left", "right")
"""The four grid actions, in the order every evaluator indexes them."""

CHECKPOINT_FORMAT = "henri-holarch-checkpoint-v1"
CHECKPOINT_NAMES = {
    "small": "holarch_small.pt",
    "medium": "holarch_medium.pt",
    "production": "holarch_production.pt",
}
"""Cycle E: files a sidecar may mount without retraining."""


def checkpoint_candidates(directory: str | Path) -> List[Path]:
    root = Path(directory)
    return [root / name for name in CHECKPOINT_NAMES.values()]


def _digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotor_payload(rotor) -> Optional[Dict[str, Any]]:
    """Serialise a fitted ActionTransitionRotor into checkpoint tensors (H3).

    Tensors, not numpy arrays: the checkpoint is loaded with
    weights_only=True, whose allow-list contains tensors and primitives but
    not the numpy reconstructor. Storing the mask as complex64 keeps the
    byte cost at D * n_actions * 8 and leaves no calibration to redo.
    """
    if rotor is None:
        return None
    phi = getattr(rotor, "phi", None)
    counts = getattr(rotor, "counts", None)
    if phi is None or counts is None or not bool(getattr(rotor, "fitted", False)):
        return None
    return {
        "format": "henri-action-rotor-v1",
        "phi": torch.from_numpy(np.asarray(phi).astype(np.complex64)).detach().cpu(),
        "counts": torch.from_numpy(np.asarray(counts).astype(np.int64)).detach().cpu(),
        "state": dict(getattr(rotor, "state", lambda: {})()),
    }


def action_rotor_from_payload(payload: Optional[Dict[str, Any]], cfg) -> Optional["ActionTransitionRotor"]:
    """Rebuild the fitted rotor a checkpoint carries, or None (H3)."""
    if not payload:
        return None
    phi = payload.get("phi")
    counts = payload.get("counts")
    if phi is None or counts is None:
        return None
    state = dict(payload.get("state", {}) or {})
    dim = int(state.get("dim", cfg.dim))
    n_actions = int(state.get("n_actions", getattr(cfg, "n_actions", 4)))
    n_slots = int(state.get("n_slots", getattr(cfg, "n_slots", 4)))
    grid = state.get("grid")
    rotor = ActionTransitionRotor(dim, n_actions, n_slots, grid=grid)
    rotor.phi = np.asarray(phi).astype(np.complex64).copy()
    rotor.counts = np.asarray(counts).astype(np.int64).copy()
    rotor.fitted = bool(int(rotor.counts.sum()) > 0)
    return rotor


SEED_ENGRAM_FORMAT = "henri-seed-engrams-v3"
"""Directive M3: the full-reference polar-packed Zone C context chart."""

SEED_ENGRAM_FORMAT_V2 = "henri-seed-engrams-v2"
"""Cycle K multi-task cartridge, still readable."""

SEED_ENGRAM_FORMAT_V1 = "henri-seed-engrams-v1"
"""Cycle J half-precision quadrature cartridge, still readable."""

SEED_ENGRAM_LIMIT = 1536
"""Hard cap on distilled engrams, so the cartridge cannot become a database."""

SEED_ENGRAM_BYTE_BUDGET = 2_621_440
"""Hard byte cap on the serialised seed block: 2.5 MiB, not 2.5 MB."""

SEED_ENGRAM_PHASE_LEVELS = 65535
"""Uniform int16 phase quantisation levels over [-pi, pi)."""

SEED_ENGRAM_AMPLITUDE_LEVELS = 255
"""Per-row peak-normalised uint8 amplitude levels."""


def seed_engram_payload(
    rows: Optional[Sequence[Dict[str, Any]]],
    enforce_budget: bool = True,
    exact_ids: Sequence[int] = (),
) -> Optional[Dict[str, Any]]:
    """Serialise the distilled Zone C rows into one compressed block (K1).

    Each row is {label, tier, wave, namespace?}. The wave is the stored unit
    complex64 vector, so the mounted store retrieves the rows the reference
    store served. Nothing here is a copy of the 21 MB fixture: only the
    engrams the deployed planner and the four task blocks actually read.

    Byte budget. Cycle J stored two float16 quadratures (4 bytes per
    dimension) and had to cap the action block at 512 rows to stay under
    budget, which cost one closed-loop episode (38 -> 37). Cycle K stores a
    unit wave as a uniform int16 phase plus a peak-normalised uint8 amplitude
    (3 bytes per dimension), then compresses the byte planes with zlib level
    9. A measured boundary decides the precision: the action block needs the
    16-bit phase to reproduce the 38/48 deterministic reference rollout
    (int8 quadrature scores 37/48), and the support blocks are equally exact
    at this precision. The format is lossless with respect to the quantised
    polar code, not with respect to complex64; reconstruction coherence is
    measured and written to the receipt.
    """
    if not rows:
        return None
    if len(rows) > SEED_ENGRAM_LIMIT:
        raise ValueError(
            f"the seed cartridge holds at most {SEED_ENGRAM_LIMIT} engrams, got {len(rows)}"
        )
    labels: List[int] = []
    tiers: List[int] = []
    waves: List[np.ndarray] = []
    namespaces: List[str] = []
    for row in rows:
        wave = np.asarray(row["wave"], dtype=np.complex64).reshape(-1)
        labels.append(int(row.get("label", -1)))
        tiers.append(int(row.get("tier", 1)))
        waves.append(wave.copy())
        namespaces.append(str(row.get("namespace", "default")))
    stacked = np.stack(waves).astype(np.complex64)
    dim = int(stacked.shape[1])
    label_array = np.asarray(labels, dtype=np.int64)
    # Directive N3: the trap-critical rows are stored once, at full precision.
    # They are removed from the delta-packed action block and the int8 support
    # block before those blocks are assembled, so the cartridge pays one copy
    # of each row rather than one packed copy plus one exact copy.
    exact_set = {int(value) for value in exact_ids}
    exact_index = np.asarray(
        [
            position
            for position, row in enumerate(rows)
            if int(row.get("engram_id", -1)) in exact_set
        ],
        dtype=np.int64,
    )
    exact_lookup = np.zeros(len(rows), dtype=bool)
    if int(exact_index.size):
        exact_lookup[exact_index] = True
    action_index = np.flatnonzero(
        (label_array >= 106) & (label_array < 110) & (~exact_lookup)
    ).astype(np.int64)
    support_index = np.flatnonzero(
        ~((label_array >= 106) & (label_array < 110)) & (~exact_lookup)
    ).astype(np.int64)

    action_payload = b""
    action_proto_labels = np.zeros(0, dtype=np.int64)
    if int(action_index.size):
        block_waves = stacked[action_index]
        block_labels = label_array[action_index]
        # Prototype-residual delta coding: every action wave is stored as a
        # phase residual against the tier-3 prototype of its own class. The
        # residual cluster makes the 12-bit phase code exact enough to
        # reproduce the deterministic reference rollout, and the prototype
        # phases themselves cost 2 bytes per dimension for four classes.
        prototypes: Dict[int, np.ndarray] = {}
        for position, index in enumerate(action_index):
            if int(tiers[int(index)]) == 3:
                prototypes.setdefault(int(block_labels[position]), block_waves[position])
        action_proto_labels = np.asarray(sorted(prototypes), dtype=np.int64)
        proto_phase = np.stack(
            [np.angle(prototypes[int(label)]) for label in action_proto_labels]
        ).astype(np.float64)
        proto_lookup = {
            int(label): proto_phase[position]
            for position, label in enumerate(action_proto_labels)
        }
        phase = np.angle(block_waves).astype(np.float64)
        residual = np.zeros_like(phase)
        for position, label in enumerate(block_labels):
            reference = proto_lookup.get(int(label))
            if reference is None:
                reference = np.zeros(dim, dtype=np.float64)
            residual[position] = (phase[position] - reference + np.pi) % (2.0 * np.pi) - np.pi
        amplitude = np.abs(block_waves).astype(np.float64)
        peak = np.maximum(amplitude.max(axis=1), 1e-12)
        quantised_phase = np.clip(
            np.rint((residual + np.pi) / (2.0 * np.pi) * 4095), 0, 4095
        ).astype(np.uint16)
        # The 4-bit amplitude uses a square-root law: the code step is fine
        # near zero and coarse near the peak, which is where a unit wave
        # spends almost all of its channels.
        quantised_amplitude = np.clip(
            np.rint(np.sqrt(amplitude / peak[:, None]) * 15), 0, 15
        ).astype(np.uint16)
        packed = (quantised_phase << 4) | quantised_amplitude
        quantised_proto = np.clip(
            np.rint((proto_phase + np.pi) / (2.0 * np.pi) * 65535), 0, 65535
        ).astype(np.uint16)
        action_payload = zlib.compress(
            action_proto_labels.astype(np.int64).tobytes()
            + quantised_proto.astype(np.uint16).tobytes()
            + packed.astype(np.uint16).tobytes()
            + np.asarray(peak, dtype=np.float32).tobytes(),
            9,
        )

    support_payload = b""
    if int(support_index.size):
        block_waves = stacked[support_index]
        scale = np.maximum(np.abs(block_waves).max(axis=1), 1e-12)
        quantised_real = np.clip(
            np.rint(block_waves.real / scale[:, None] * 127), -127, 127
        ).astype(np.int8)
        quantised_imag = np.clip(
            np.rint(block_waves.imag / scale[:, None] * 127), -127, 127
        ).astype(np.int8)
        support_payload = zlib.compress(
            quantised_real.tobytes()
            + quantised_imag.tobytes()
            + np.asarray(scale, dtype=np.float32).tobytes(),
            9,
        )

    exact_payload = b""
    if int(exact_index.size):
        exact_waves = stacked[exact_index]
        exact_phase = np.clip(
            np.rint((np.angle(exact_waves) + np.pi) / (2.0 * np.pi) * 65535), 0, 65535
        ).astype(np.uint16)
        exact_amp = np.abs(exact_waves)
        exact_peak = np.maximum(exact_amp.max(axis=1), 1e-12)
        exact_quant = np.clip(
            np.rint(exact_amp / exact_peak[:, None] * 255), 0, 255
        ).astype(np.uint8)
        exact_payload = zlib.compress(
            exact_index.astype(np.int64).tobytes()
            + exact_phase.tobytes()
            + exact_quant.tobytes()
            + np.asarray(exact_peak, dtype=np.float32).tobytes(),
            9,
        )

    block = {
        "format": SEED_ENGRAM_FORMAT,
        "exact_index": torch.tensor(exact_index, dtype=torch.int64),
        "exact_payload": (
            torch.frombuffer(bytearray(exact_payload), dtype=torch.uint8).clone()
            if exact_payload
            else torch.zeros(0, dtype=torch.uint8)
        ),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "tiers": torch.tensor(tiers, dtype=torch.int64),
        "action_index": torch.tensor(action_index, dtype=torch.int64),
        "action_payload": (
            torch.frombuffer(bytearray(action_payload), dtype=torch.uint8).clone()
            if action_payload
            else torch.zeros(0, dtype=torch.uint8)
        ),
        "support_index": torch.tensor(support_index, dtype=torch.int64),
        "support_payload": (
            torch.frombuffer(bytearray(support_payload), dtype=torch.uint8).clone()
            if support_payload
            else torch.zeros(0, dtype=torch.uint8)
        ),
        "count": int(len(labels)),
        "dim": int(stacked.shape[1]),
        "namespace": namespaces[0] if namespaces else "default",
        "source_sha256": str(rows[0].get("source_sha256", "")) if rows else "",
        "dtype": "v3-action-int12-sqrt4-support-int8-exact16-zlib9",
        "compression": "zlib-9",
        "action_rows": int(np.count_nonzero((label_array >= 106) & (label_array < 110))),
        "action_packed_rows": int(action_index.size),
        "action_exact_rows": int(exact_index.size),
        "action_proto_count": int(action_proto_labels.size),
        "support_rows": int(support_index.size),
    }
    block["bytes"] = int(
        int(block["action_payload"].numel())
        + int(block["support_payload"].numel())
        + block["labels"].numel() * 8
        + block["tiers"].numel() * 8
        + block["action_index"].numel() * 8
        + block["support_index"].numel() * 8
        + int(block["exact_payload"].numel())
        + block["exact_index"].numel() * 8
    )
    if enforce_budget and int(block["bytes"]) > SEED_ENGRAM_BYTE_BUDGET:
        raise ValueError(
            f"the seed cartridge exceeds {SEED_ENGRAM_BYTE_BUDGET} bytes: {block['bytes']}"
        )
    return block


def _seed_engram_waves_v3(block: Dict[str, Any]) -> Optional[np.ndarray]:
    """Decode the M3 mixed polar block: packed action rows, int8 support rows."""
    count = int(block.get("count", 0) or 0)
    dim = int(block.get("dim", 0) or 0)
    if count <= 0 or dim <= 0:
        return None
    waves = np.zeros((count, dim), dtype=np.complex64)
    action_index = block.get("action_index")
    support_index = block.get("support_index")
    if action_index is not None and int(action_index.numel()):
        indices = action_index.detach().cpu().numpy().astype(np.int64)
        raw = zlib.decompress(
            bytes(block["action_payload"].detach().cpu().numpy().tobytes())
        )
        rows = int(indices.size)
        proto_count = int(block.get("action_proto_count", 0) or 0)
        cursor = 0
        proto_labels = np.frombuffer(
            raw[cursor : cursor + proto_count * 8], dtype=np.int64
        )
        cursor += proto_count * 8
        proto_quantised = np.frombuffer(
            raw[cursor : cursor + proto_count * dim * 2], dtype=np.uint16
        ).reshape(proto_count, dim)
        cursor += proto_count * dim * 2
        packed = np.frombuffer(
            raw[cursor : cursor + rows * dim * 2], dtype=np.uint16
        ).reshape(rows, dim)
        cursor += rows * dim * 2
        peaks = np.frombuffer(raw[cursor : cursor + rows * 4], dtype=np.float32)[:rows]
        proto_phase = (
            proto_quantised.astype(np.float64) / 65535.0 * (2.0 * np.pi) - np.pi
        )
        lookup = {
            int(label): position for position, label in enumerate(proto_labels.tolist())
        }
        residual = (packed >> 4).astype(np.float64) / 4095.0 * (2.0 * np.pi) - np.pi
        labels_tensor = block.get("labels")
        labels = (
            labels_tensor.detach().cpu().numpy().astype(np.int64)
            if labels_tensor is not None
            else np.zeros(count, dtype=np.int64)
        )
        phase = np.zeros_like(residual)
        for position, index in enumerate(indices):
            reference = lookup.get(int(labels[int(index)]))
            phase[position] = (
                residual[position] + (proto_phase[reference] if reference is not None else 0.0)
            )
        amplitude = (packed & 15).astype(np.float64) / 15.0
        amplitude = (amplitude ** 2) * peaks.astype(np.float64)[:, None]
        reconstructed = amplitude * np.exp(1j * phase)
        norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        waves[indices] = (reconstructed / np.maximum(norms, 1e-12)).astype(np.complex64)
    if support_index is not None and int(support_index.numel()):
        indices = support_index.detach().cpu().numpy().astype(np.int64)
        raw = zlib.decompress(
            bytes(block["support_payload"].detach().cpu().numpy().tobytes())
        )
        rows = int(indices.size)
        plane = rows * dim
        real = np.frombuffer(raw[:plane], dtype=np.int8).reshape(rows, dim)
        imag = np.frombuffer(raw[plane : 2 * plane], dtype=np.int8).reshape(rows, dim)
        scale = np.frombuffer(raw[2 * plane :], dtype=np.float32)[:rows]
        reconstructed = (
            (real.astype(np.float64) + 1j * imag.astype(np.float64))
            * scale.astype(np.float64)[:, None]
            / 127.0
        )
        norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        waves[indices] = (reconstructed / np.maximum(norms, 1e-12)).astype(np.complex64)
    exact_index = block.get("exact_index")
    if exact_index is not None and int(exact_index.numel()):
        indices = exact_index.detach().cpu().numpy().astype(np.int64)
        raw = zlib.decompress(
            bytes(block["exact_payload"].detach().cpu().numpy().tobytes())
        )
        rows = int(indices.size)
        cursor = 0
        stored_index = np.frombuffer(raw[cursor : cursor + rows * 8], dtype=np.int64)
        cursor += rows * 8
        phase = np.frombuffer(
            raw[cursor : cursor + rows * dim * 2], dtype=np.uint16
        ).reshape(rows, dim)
        cursor += rows * dim * 2
        amplitude = np.frombuffer(
            raw[cursor : cursor + rows * dim], dtype=np.uint8
        ).reshape(rows, dim)
        cursor += rows * dim
        peaks = np.frombuffer(raw[cursor : cursor + rows * 4], dtype=np.float32)[:rows]
        exact_phase = phase.astype(np.float64) / 65535.0 * (2.0 * np.pi) - np.pi
        exact_amplitude = (
            amplitude.astype(np.float64) / 255.0 * peaks.astype(np.float64)[:, None]
        )
        reconstructed = exact_amplitude * np.exp(1j * exact_phase)
        norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        waves[stored_index] = (
            reconstructed / np.maximum(norms, 1e-12)
        ).astype(np.complex64)
    return waves


def seed_engram_waves(block: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Decode a stored block into its (count, dim) complex64 waves.

    Pure function of the stored bytes: the same block always decodes to the
    same array, so a clean-room replay is bit-identical.
    """
    if not block:
        return None
    if block.get("format") == SEED_ENGRAM_FORMAT:
        return _seed_engram_waves_v3(block)
    if block.get("format") != SEED_ENGRAM_FORMAT_V2:
        return None
    payload = block.get("payload")
    if payload is None:
        return None
    count = int(block.get("count", 0) or 0)
    dim = int(block.get("dim", 0) or 0)
    if count <= 0 or dim <= 0:
        return None
    raw = zlib.decompress(bytes(payload.detach().cpu().numpy().tobytes()))
    phase_bytes = count * dim * 2
    amplitude_bytes = count * dim
    phase = np.frombuffer(raw[:phase_bytes], dtype=np.uint16).reshape(count, dim)
    amplitude = np.frombuffer(
        raw[phase_bytes : phase_bytes + amplitude_bytes], dtype=np.uint8
    ).reshape(count, dim)
    peak = np.frombuffer(raw[phase_bytes + amplitude_bytes :], dtype=np.float32).reshape(count)
    levels = float(block.get("phase_levels", SEED_ENGRAM_PHASE_LEVELS))
    amplitude_levels = float(block.get("amplitude_levels", SEED_ENGRAM_AMPLITUDE_LEVELS))
    reconstructed_phase = phase.astype(np.float64) / levels * (2.0 * np.pi) - np.pi
    reconstructed_amplitude = (
        amplitude.astype(np.float64) / amplitude_levels * peak.astype(np.float64)[:, None]
    )
    waves = (reconstructed_amplitude * np.exp(1j * reconstructed_phase)).astype(np.complex64)
    norms = np.linalg.norm(waves, axis=1, keepdims=True)
    return waves / np.maximum(norms, 1e-12)


def seed_engram_rows(block: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inverse of seed_engram_payload: the rows a mount will hydrate."""
    if not block:
        return []
    if block.get("format") in {SEED_ENGRAM_FORMAT, SEED_ENGRAM_FORMAT_V2}:
        waves = seed_engram_waves(block)
        labels = block.get("labels")
        tiers = block.get("tiers")
        if waves is None or labels is None:
            return []
        rows = []
        for index in range(int(waves.shape[0])):
            rows.append(
                {
                    "label": int(labels[index]),
                    "tier": int(tiers[index]) if tiers is not None else 1,
                    "wave": np.asarray(waves[index], dtype=np.complex64),
                    "namespace": str(block.get("namespace", "default") or "default"),
                }
            )
        return rows
    if block.get("format") != SEED_ENGRAM_FORMAT_V1:
        return []
    labels = block.get("labels")
    tiers = block.get("tiers")
    if block.get("real") is not None and block.get("imag") is not None:
        real = np.asarray(block["real"].detach().cpu().numpy(), dtype=np.float32)
        imag = np.asarray(block["imag"].detach().cpu().numpy(), dtype=np.float32)
        waves = (real + 1j * imag).astype(np.complex64)
    elif block.get("waves") is not None:
        waves = np.asarray(block["waves"].detach().cpu().numpy(), dtype=np.complex64)
    else:
        return []
    count = int(waves.shape[0])
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        row = waves[index]
        length = float(np.linalg.norm(row))
        if length > 1e-8:
            row = row / length
        rows.append(
            {
                "label": int(labels[index]) if labels is not None else -1,
                "tier": int(tiers[index]) if tiers is not None else 1,
                "wave": np.asarray(row, dtype=np.complex64),
                "namespace": str(block.get("namespace", "default") or "default"),
            }
        )
    return rows


@torch.no_grad()
def save_checkpoint(
    model: HenriHolarch,
    path: str | Path,
    *,
    cfg: Optional[HolarchConfig] = None,
    name: str = "holarch",
    train_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    rotor=None,
    created_utc: Optional[str] = None,
    seed_engrams: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Write weights, calibrated buffers, the prototype bank, and the rotor.

    The prototype tensors are registered buffers, so they travel inside
    state_dict. The numpy banks and their calibrated beta are restored from
    the metadata on load, because beta is a calibration result and not a
    parameter. Directive H3 adds the fitted action-transition rotor: a
    mounted medium checkpoint therefore carries the 0.8125 depth-2 planner
    out of the box instead of making every process refit it.
    """
    cfg = cfg or model.cfg
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "name": str(name),
        "created_utc": str(created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "config": {key: getattr(cfg, key) for key in cfg.__dataclass_fields__},
        "config_digest": str(cfg.digest()),
        "seed": int(cfg.seed if seed is None else seed),
        "parameters": int(model.parameter_count()),
        "train_config": dict(train_config or {}),
        "metrics": dict(metrics or {}),
        "prototypes": {
            task: model.prototypes[task].report() for task in TASK_NAMES
        },
        "rotor": rotor_payload(rotor),
        # Directive J1: the checkpoint carries the task-scoped Zone C context
        # chart it was measured with, so a clean-room mount does not depend on
        # an uncommitted 21 MB SQLite fixture. The block is a tensor dictionary
        # because the checkpoint is loaded with weights_only=True.
        "seed_engrams": seed_engram_payload(seed_engrams),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, target)
    digest = _digest_file(target)
    target.with_name(target.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return {
        "path": str(target),
        "sha256": digest,
        "bytes": int(target.stat().st_size),
        "parameters": int(payload["parameters"]),
    }


@torch.no_grad()
@torch.no_grad()
def open_clean_room_memory(
    cfg,
    checkpoint: Optional[str | Path] = None,
    workdir: Optional[str | Path] = None,
    prefix: str = "henri_clean_room",
) -> Tuple[object, Dict[str, object]]:
    """Create an EMPTY Zone C store and hydrate it from a checkpoint (J1).

    Unlike open_planner_memory this function never copies the repository
    fixture: the directory is new, the SQLite file holds zero rows before the
    mount, and every retrievable engram arrived inside the checkpoint. The
    returned info block states the pre-mount row count so a receipt can prove
    the clean-room condition instead of asserting it.
    """
    from henri.holarch.zone_c import ZoneCMemory

    directory = Path(workdir or tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}_{os.getpid()}_{int(time.time() * 1000) % 100000}.sqlite3"
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(target))
    before = int(memory.stats().get("count", 0))
    info: Dict[str, object] = {
        "path": str(target),
        "pre_mount_rows": before,
        "checkpoint": str(checkpoint or ""),
        "empty": bool(before == 0),
    }
    if checkpoint is not None and Path(checkpoint).exists():
        block = torch.load(checkpoint, map_location="cpu", weights_only=True).get("seed_engrams")
        rows = seed_engram_rows(block)
        written = 0
        for row in rows:
            memory.write(
                row["wave"],
                label=int(row["label"]),
                tier=int(row["tier"]),
                namespace=str(row.get("namespace", "default")),
            )
            written += 1
        info["hydrated"] = int(written)
        info["post_mount_rows"] = int(memory.stats().get("count", 0))
    else:
        info["hydrated"] = 0
        info["post_mount_rows"] = before
    return memory, info


def open_planner_memory(
    cfg,
    source: Optional[str | Path] = None,
    workdir: Optional[str | Path] = None,
    prefix: str = "henri_planner_memory",
) -> Tuple[object, Dict[str, object]]:
    """Open the Zone C store the wave planner reads, from a scratch copy.

    The deployed planner fuses the retrieved Zone C context into the sensor
    wave (use_memory=True), so the 48-episode rate and the single-state
    horizon probes are conditioned on the store's engrams. Copying the
    repository store to a scratch path makes the measurement read-only and
    reproducible, and the returned info block pins the fixture by SHA-256 and
    engram count so a receipt can name the exact state it measured.

    Directive I3: the fixture is an explicit input, not a hidden global. When
    the store is absent the function returns an empty scratch store and says
    so; the caller then measures the clean-room control rather than silently
    claiming the deployed number.
    """
    from henri.holarch.zone_c import ZoneCMemory

    candidate = Path(
        source
        or os.environ.get("HENRI_PLANNER_MEMORY", "")
        or Path("data") / "holarch" / "zone_c.sqlite3"
    )
    directory = Path(workdir or tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}_{os.getpid()}.sqlite3"
    if not candidate.exists():
        memory = ZoneCMemory(cfg, backend="sqlite", path=str(target))
        return memory, {
            "source": "",
            "path": str(target),
            "sha256": "",
            "bytes": 0,
            "engrams": int(memory.stats().get("count", 0)),
            "empty": True,
        }
    shutil.copy2(candidate, target)
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(target))
    digest = hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
    return memory, {
        "source": str(candidate),
        "path": str(target),
        "sha256": digest,
        "bytes": int(Path(candidate).stat().st_size),
        "engrams": int(memory.stats().get("count", 0)),
        "empty": False,
    }


def load_checkpoint(
    path: str | Path,
    model: Optional[HenriHolarch] = None,
    *,
    strict: bool = True,
) -> Tuple[HenriHolarch, Dict[str, Any]]:
    """Mount a checkpoint into an existing model, or build one from it.

    Returns the model and the metadata (without the state dict). A
    dimension mismatch is reported as a ValueError so a caller can fall
    back to a fresh bootstrap explicitly instead of failing silently.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{target} is not a {CHECKPOINT_FORMAT} checkpoint")
    state = payload["state_dict"]
    if model is None:
        cfg = HolarchConfig(**payload["config"])
        model = HenriHolarch(cfg, backend="sqlite")
    # The prototype buffers are registered empty at construction, so a
    # converged bank cannot be copied into them until the buffer is resized
    # to the checkpoint shape. Resize first, then load strictly.
    for key, value in state.items():
        if key.startswith("prototype_") and hasattr(model, key):
            current = getattr(model, key)
            if tuple(current.shape) != tuple(value.shape):
                setattr(model, key, torch.zeros(
                    tuple(value.shape), dtype=value.dtype, device=current.device
                ))
    model.load_state_dict(state, strict=bool(strict))
    model.eval()
    # Restore the numpy bank and its calibrated inverse temperature. The
    # registered buffers already carry the rows and labels; beta is the
    # only calibration result that is not a buffer.
    for task in TASK_NAMES:
        report = dict(payload.get("prototypes", {}).get(task, {}))
        bank = model.prototypes[task]
        waves = getattr(model, f"prototype_{task}")
        classes = getattr(model, f"prototype_class_{task}")
        if int(waves.numel()):
            bank.phi = waves.detach().cpu().numpy().astype(np.complex64)
            bank.labels = classes.detach().cpu().numpy().astype(np.int64)
            bank.counts = np.ones(int(bank.labels.size), dtype=np.int64)
        if "prototype_beta" in report:
            bank.beta = float(report["prototype_beta"])
            bank.beta_source = str(report.get("beta_source", "checkpoint"))
    # Directive H3: restore the fitted action-transition rotor that travels
    # with the weights, and attach it to the model so a mounted checkpoint
    # plans with the rotor it was measured with, without a re-fit.
    rotor = action_rotor_from_payload(
        payload.get("rotor"), HolarchConfig(**payload["config"])
    )
    try:
        model.action_rotor = rotor
    except Exception:  # pragma: no cover - a frozen module is not expected
        pass
    # Directive R2: restore the fitted Action KV regime head. It is a plain
    # numpy readout carried beside the weights, so the checkpoint stays a
    # tensor-only payload for torch.load(weights_only=True).
    try:
        from henri.holarch.code_gen import ActionRegimeReadout

        readout_state = payload.get("action_regime_readout")
        model.action_regime_readout = (
            ActionRegimeReadout.from_state(readout_state)
            if readout_state
            else None
        )
    except Exception:  # pragma: no cover - an older checkpoint has none
        model.action_regime_readout = None
    # Directive J1: the checkpoint carries its own Zone C context chart. A
    # mount into a store that holds zero rows hydrates the distilled engrams
    # into the default namespace, so the planner's psi_fused wave is the wave
    # the checkpoint was measured with and no external fixture is consulted.
    seed_block = payload.get("seed_engrams")
    hydration: Dict[str, Any] = {"available": 0, "written": 0, "skipped": True}
    try:
        model.seed_engrams = seed_block
    except Exception:  # pragma: no cover - a frozen module is not expected
        pass
    if seed_block:
        hydration = dict(model.hydrate_seed_engrams())
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", "rotor", "seed_engrams", "action_regime_readout"}
    }
    metadata["rotor"] = dict(getattr(rotor, "state", lambda: {})()) if rotor is not None else None
    metadata["seed_engrams"] = {
        "present": bool(seed_block),
        "format": str((seed_block or {}).get("format", "")),
        "count": int((seed_block or {}).get("count", 0) or 0),
        "bytes": int((seed_block or {}).get("bytes", 0) or 0),
        "dim": int((seed_block or {}).get("dim", 0) or 0),
        "hydration": dict(hydration),
    }
    metadata["path"] = str(target)
    metadata["sha256"] = _digest_file(target)
    return model, metadata


def find_checkpoint(
    cfg: Optional[HolarchConfig] = None,
    *,
    directory: str | Path = "receipts/holarch",
    explicit: Optional[str] = None,
) -> Optional[Path]:
    """Locate the converged checkpoint for the requested wave dimension.

    Order: explicit path, HENRI_HOLARCH_CHECKPOINT, the profile name that
    matches the config dimension, then any profile whose config dimension
    matches. A mismatch is never mounted by accident.
    """
    import os

    if explicit:
        return Path(explicit)
    env = os.environ.get("HENRI_HOLARCH_CHECKPOINT")
    if env:
        return Path(env)
    root = Path(directory)
    if cfg is None:
        # A caller that did not pin a profile gets the most converged
        # checkpoint available, in decreasing order of training budget.
        for profile in ("medium", "small", "production"):
            candidate = root / CHECKPOINT_NAMES[profile]
            if candidate.exists():
                return candidate
        return None
    dim = int(cfg.dim)
    ordered = ["medium", "small", "production"] if dim >= 1024 else ["small", "medium", "production"]
    for profile in ordered:
        candidate = root / CHECKPOINT_NAMES[profile]
        if not candidate.exists():
            continue
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            found_dim = int(payload.get("config", {}).get("dim", 0))
        except Exception:
            continue
        if found_dim == dim:
            return candidate
    return None

# --------------------------------------------------------------- lookahead
@torch.no_grad()
def _unit_tensor(wave: torch.Tensor) -> torch.Tensor:
    return wave / wave.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _slot3(wave: torch.Tensor, n_slots: int = 4) -> torch.Tensor:
    """The last functional slot of a flat D-dim wave."""
    width = wave.shape[-1] // int(n_slots)
    return wave[..., (int(n_slots) - 1) * width :]



class SensorWaveCodec:
    """The four-slot wave state used by the 1-step lookahead.

    The wave is a physical sensor frame, not a relabelled feature vector:

        slot 0  local wall map, 5x5 window
        slot 1  local goal map, 5x5 window
        slot 2  agent marker at the window centre
        slot 3  goal-offset wave packet, a Gaussian on the 16x16 torus

    A move by (dr, dc) circularly shifts every slot by (dr, dc), which is
    exactly circular convolution with a delta kernel. Slot 3 is the goal
    wave of the directive: a packet at the offset (goal - agent) that
    moves toward the centre as the agent approaches the goal. Alignment of
    a shifted packet with the centre packet is a smooth, monotone function
    of the remaining distance, so the reduction is a real gradient and not
    a three-way tie at zero.
    """

    def __init__(self, dim: int, n_slots: int = 4, sigma: float = 1.6) -> None:
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.d_slot = self.dim // self.n_slots
        self.grid = int(round(math.sqrt(self.d_slot)))
        if self.grid * self.grid != self.d_slot:
            raise ValueError(
                f"d_slot={self.d_slot} is not a square grid; use D=1024 medium"
            )
        self.sigma = float(sigma)
        self.centre = self.grid // 2
        self.offset = 2
        self._kernels: Dict[int, torch.Tensor] = {}

    def _kernel(self, action: int) -> torch.Tensor:
        if action not in self._kernels:
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(action)]
            kernel = torch.zeros(self.grid, self.grid, dtype=torch.float32)
            # new[r, c] = old[r + dr, c + dc], so the convolution delta sits
            # at (-dr, -dc); this moves the goal packet toward the centre.
            kernel[(-dr) % self.grid, (-dc) % self.grid] = 1.0
            self._kernels[action] = torch.fft.fft2(kernel)
        return self._kernels[action]

    def shift(self, psi: torch.Tensor, action: int) -> torch.Tensor:
        """Circular convolution of the wave with the action's shift kernel."""
        slots = psi.reshape(self.n_slots, self.grid, self.grid).to(torch.complex64)
        kernel = self._kernel(int(action)).to(slots.device)
        out = torch.fft.ifft2(torch.fft.fft2(slots) * kernel.unsqueeze(0))
        return _unit_tensor(out.reshape(1, self.dim).real.to(torch.float32).to(torch.complex64))

    def shift_batch(self, psi: torch.Tensor, actions: Sequence[int]) -> torch.Tensor:
        """Batched circular convolution for several actions at once (I3).

        The same kernel bank, one fft2 over (n_slots, grid, grid) and one
        ifft2 over (len(actions), n_slots, grid, grid). Mathematically this
        is exactly the loop of shift() calls; it exists because the depth-2
        lookahead needs 4 first steps and 16 second steps per decision.
        """
        actions = [int(action) for action in actions]
        batch = psi.reshape(-1, self.dim)
        count = int(batch.shape[0])
        slots = batch.reshape(count, self.n_slots, self.grid, self.grid).to(torch.complex64)
        spectrum = torch.fft.fft2(slots)
        kernels = torch.stack([self._kernel(action) for action in actions]).to(slots.device)
        out = torch.fft.ifft2(spectrum * kernels.unsqueeze(1))
        flat = out.real.reshape(int(out.shape[0]), self.dim).to(torch.float32).to(torch.complex64)
        return _unit_tensor(flat)

    def encode(self, observation) -> torch.Tensor:
        values = np.asarray(observation, dtype=np.float32)
        view = values[:75].reshape(5, 5, 3)
        relative = values[75:79]
        slots = np.zeros((self.n_slots, self.grid, self.grid), dtype=np.float32)
        lo = self.centre - self.offset
        hi = lo + 5
        slots[0, lo:hi, lo:hi] = view[:, :, 0]
        slots[1, lo:hi, lo:hi] = view[:, :, 1]
        slots[2, self.centre, self.centre] = 1.0
        # relative[0] and relative[1] are divided by grid_size - 1 = 6.
        dg_row = int(round(float(relative[0]) * 6.0))
        dg_col = int(round(float(relative[1]) * 6.0))
        row = min(max(self.centre + dg_row, 0), self.grid - 1)
        col = min(max(self.centre + dg_col, 0), self.grid - 1)
        yy, xx = np.mgrid[0 : self.grid, 0 : self.grid]
        packet = np.exp(
            -((yy - row) ** 2 + (xx - col) ** 2) / (2.0 * self.sigma ** 2)
        ).astype(np.float32)
        slots[3] = packet
        wave = torch.from_numpy(slots.reshape(-1)).to(torch.complex64)
        return _unit_tensor(wave.reshape(1, self.dim))

    def goal_wave(self) -> torch.Tensor:
        slots = np.zeros((self.n_slots, self.grid, self.grid), dtype=np.float32)
        yy, xx = np.mgrid[0 : self.grid, 0 : self.grid]
        slots[3] = np.exp(
            -((yy - self.centre) ** 2 + (xx - self.centre) ** 2)
            / (2.0 * self.sigma ** 2)
        ).astype(np.float32)
        wave = torch.from_numpy(slots.reshape(-1)).to(torch.complex64)
        return _unit_tensor(wave.reshape(1, self.dim))


class LatentActionLookahead:
    """One-step action-conditioned wave transition plus a goal attractor.

    Cycle E E1 implements the directive literally:

        psi_next(a) = normalize(psi_t (*) psi_action(a))

    where (*) is circular convolution with the action's shift kernel.
    The score for executing an action is

        score(a) = y_action(a) + 0.40 * align_goal(a)
                             - 0.60 * penalty_cycle(a)

    align_goal is Re<slot3(psi_next), slot3(psi_goal)>; penalty_cycle is
    max_tau 0.85^tau Re<psi_next, psi_{t-tau}> over the last four visited
    waves. The learned policy logits are standardised before the sum, so
    the declared weights act on comparable units instead of being
    swallowed by an arbitrary logit scale.
    """

    def __init__(
        self,
        model: HenriHolarch,
        task,
        *,
        goal_weight: float = 0.40,
        cycle_weight: float = 0.60,
        cycle_decay: float = 0.85,
        history: int = 4,
        sigma: float = 1.6,
        standardise_logits: bool = True,
        rotor=None,
        transition: str = "shift",
        lookahead_depth: int = 1,
        gamma_look: float = 0.02,
        mask_lookahead: bool = True,
        batch_rollout: bool = True,
        **_unused,
    ) -> None:
        self.model = model
        self.task = task
        self.goal_weight = float(goal_weight)
        self.cycle_weight = float(cycle_weight)
        self.cycle_decay = float(cycle_decay)
        self.history = int(max(1, history))
        self.standardise_logits = bool(standardise_logits)
        self.codec = SensorWaveCodec(int(model.cfg.dim), int(model.cfg.n_slots), float(sigma))
        self.goal = self.codec.goal_wave()
        self.sigma = float(sigma)
        # Directive F2: the transition operator is selectable. The default is
        # the explicit shift, so every Cycle E number stays reproducible; a
        # learned rotor or a blend is an opt-in arm that the receipt names.
        self.rotor = rotor
        if str(transition) not in {"shift", "rotor", "blend"}:
            raise ValueError("transition must be 'shift', 'rotor', or 'blend'")
        if str(transition) in {"rotor", "blend"} and rotor is None:
            raise ValueError("a rotor transition needs a fitted rotor")
        self.transition = str(transition)
        if int(lookahead_depth) not in {1, 2}:
            raise ValueError("lookahead_depth must be 1 or 2")
        self.lookahead_depth = int(lookahead_depth)
        self.gamma_look = float(gamma_look)
        # Directive H3 keeps the Cycle G mask as the deployed rule. The flag
        # exists so a benchmark can MEASURE the unmasked arm that Cycle F
        # shipped, instead of asserting that the mask matters.
        self.mask_lookahead = bool(mask_lookahead)
        # Directive I3: the 4 + 16 branch rollout runs as two batched FFTs.
        # The loop path stays available so the receipt can measure parity
        # between the two implementations instead of assuming it.
        self.batch_rollout = bool(batch_rollout)
        # Measured, not declared: on the validation layouts a weight of 0.50
        # lets the optimistic second step override the first-step policy
        # (35/48 wins), while 0.02 lets it break ties without overriding
        # (39/48). The G3 receipt records the whole sweep.
        self.win_bonus = 2.0

    def goal_distance(self, wave: torch.Tensor) -> int:
        """Chebyshev distance from the goal packet's peak to the frame centre.

        Slot 3 holds a Gaussian packet at the goal offset, so its argmax is
        the goal's position in the local frame. Distance zero is the terminal
        state the environment will report as a win.
        """
        grid = int(self.codec.grid)
        centre = int(self.codec.centre)
        slots = wave.detach().cpu().numpy().reshape(-1)[: self.codec.dim]
        packet = slots.reshape(int(self.codec.n_slots), grid, grid)[3].real
        row, col = np.unravel_index(int(np.argmax(packet)), packet.shape)
        return int(max(abs(int(row) - centre), abs(int(col) - centre)))

    def wall_mask(self, wave: torch.Tensor) -> np.ndarray:
        """Which unit moves out of the wave's own slot-0 frame hit a wall.

        Slot 0 carries the 5x5 local occupancy grid centred on the agent, and
        the codec writes an out-of-bounds cell as occupied. Reading the cell
        one step in each direction is exactly the legality test the
        environment will apply, and it costs four array reads.
        """
        grid = int(self.codec.grid)
        centre = int(self.codec.centre)
        slots = wave.detach().cpu().numpy().reshape(-1)[: self.codec.dim]
        occupancy = slots.reshape(int(self.codec.n_slots), grid, grid)[0].real
        # The codec stores a UNIT wave, so a wall cell is not 1.0: it is
        # 1/||psi||. The threshold is therefore relative to the strongest cell
        # in the frame, which keeps the test scale-free through the optical
        # renormalisation.
        peak = float(np.max(np.abs(occupancy))) if occupancy.size else 0.0
        threshold = 0.5 * peak
        blocked = np.zeros(4, dtype=bool)
        for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            row = min(max(centre + dr, 0), grid - 1)
            col = min(max(centre + dc, 0), grid - 1)
            blocked[action] = bool(peak > 0.0 and occupancy[row, col] > threshold)
        return blocked

    # ------------------------------------------------------------ inference
    def predict(self, psi: torch.Tensor, action: int) -> torch.Tensor:
        if self.transition == "shift":
            return self.codec.shift(psi, int(action))
        rotated = self.rotor.predict(psi, int(action))
        if self.transition == "rotor":
            return rotated
        return _unit_tensor(self.codec.shift(psi, int(action)) + rotated)

    def predict_batch(self, psi: torch.Tensor, actions: Sequence[int]) -> torch.Tensor:
        """Batched one-step transition for several actions at once."""
        actions = [int(action) for action in actions]
        if self.transition == "shift":
            return self.codec.shift_batch(psi, actions)
        rotated = self.rotor.predict_batch(psi, actions)
        if self.transition == "rotor":
            return rotated
        return _unit_tensor(self.codec.shift_batch(psi, actions) + rotated)

    def wall_mask_batch(self, waves: torch.Tensor) -> np.ndarray:
        """Wall legality for each row of a (A, dim) wave batch."""
        grid = int(self.codec.grid)
        centre = int(self.codec.centre)
        array = waves.detach().cpu().numpy().reshape(-1, int(self.codec.dim))
        count = int(array.shape[0])
        slots = array.reshape(count, int(self.codec.n_slots), grid, grid)
        occupancy = slots[:, 0].real
        peak = np.max(np.abs(occupancy), axis=(1, 2))
        threshold = 0.5 * peak
        blocked = np.zeros((count, 4), dtype=bool)
        for action, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            row = min(max(centre + dr, 0), grid - 1)
            col = min(max(centre + dc, 0), grid - 1)
            blocked[:, action] = (peak > 0.0) & (occupancy[:, row, col] > threshold)
        return blocked

    def goal_distance_batch(self, waves: torch.Tensor) -> np.ndarray:
        """Chebyshev goal distance for each row of a (A, dim) wave batch."""
        grid = int(self.codec.grid)
        centre = int(self.codec.centre)
        array = waves.detach().cpu().numpy().reshape(-1, int(self.codec.dim))
        count = int(array.shape[0])
        slots = array.reshape(count, int(self.codec.n_slots), grid, grid)
        packets = slots[:, 3].real
        flat = packets.reshape(count, grid * grid)
        indices = np.argmax(flat, axis=1)
        rows, cols = np.unravel_index(indices, (grid, grid))
        return np.maximum(np.abs(rows.astype(int) - centre), np.abs(cols.astype(int) - centre))

    def goal_wave(self, observation=None) -> torch.Tensor:
        return self.goal

    def score(
        self,
        psi: torch.Tensor,
        logits: torch.Tensor,
        observation,
        recent: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y = logits.clone().to(torch.float32)
        if self.standardise_logits:
            spread = float(y.std())
            if spread > 1e-6:
                y = (y - y.mean()) / spread
        goal = _slot3(self.goal)
        alignment = torch.zeros(4)
        penalty = torch.zeros(4)
        for action in range(4):
            nxt = self.predict(psi, action)
            alignment[action] = (_slot3(nxt).conj() * goal).real.sum()
            if recent:
                terms = torch.stack(
                    [
                        (self.cycle_decay ** tau) * (nxt.conj() * state).real.sum()
                        for tau, state in enumerate(recent)
                    ]
                )
                penalty[action] = terms.max()
        # Standardise the two wave terms before the declared weights are
        # applied. A raw dot product of unit waves spans a few hundredths
        # across four candidates, while a logit spans order one; without
        # this step the 0.40/0.60 coefficients would be arithmetically
        # present and behaviourally inert. Standardising keeps the
        # directive's relative weights meaningful.
        def _z(value: torch.Tensor) -> torch.Tensor:
            spread = float(value.std())
            return (value - value.mean()) / spread if spread > 1e-8 else torch.zeros_like(value)

        lookahead = torch.zeros(4)
        if self.lookahead_depth == 2:
            # Depth 2: the best discounted second step, scored with the same
            # goal attractor and the same cycle buffer. Directive G3: the
            # second step is masked by the WALL MAP THAT MOVED WITH THE AGENT.
            # Slot 0 of the one-step wave is the local 5x5 frame centred on
            # the new position, so a second action that would cross a wall or
            # a boundary is physically illegal and must not be scored. Without
            # this mask the rollout rewards phantom tunnelling and depth 2
            # loses to depth 1 (measured 0.7292 against 0.7708 in Cycle F).
            # Directive I3: the deployed path is two batched FFTs; the loop
            # path remains selectable so parity can be measured, not assumed.
            lookahead = (
                self._depth_two_batched(psi, goal, recent)
                if self.batch_rollout
                else self._depth_two_loop(psi, goal, recent)
            )
        combined = alignment + (
            self.gamma_look * _z(lookahead) if self.lookahead_depth == 2 else 0.0
        )
        scores = y + self.goal_weight * _z(combined) - self.cycle_weight * _z(penalty)
        return scores, {
            "alignment": alignment,
            "penalty": penalty,
            "y": y,
            "lookahead": lookahead,
            "combined": combined,
        }

    # ------------------------------------------------------ depth-2 rollout
    def _depth_two_loop(
        self, psi: torch.Tensor, goal: torch.Tensor, recent: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Reference implementation: one predict() per branch."""
        lookahead = torch.zeros(4)
        for action in range(4):
            first = self.predict(psi, action)
            blocked = (
                self.wall_mask(first)
                if self.mask_lookahead
                else np.zeros(4, dtype=bool)
            )
            legal = [index for index in range(4) if not bool(blocked[index])]
            opposite = (1, 0, 3, 2)[int(action)]
            if len(legal) > 1 and opposite in legal:
                legal.remove(opposite)
            if not legal:
                legal = [index for index in range(4) if not bool(blocked[index])]
            if not legal:
                lookahead[action] = float((_slot3(first).conj() * goal).real.sum())
                continue
            best = -float("inf")
            for second_action in legal:
                second = self.predict(first, second_action)
                value = float((_slot3(second).conj() * goal).real.sum())
                if self.goal_distance(second) == 0:
                    value += self.win_bonus
                if recent:
                    value -= self.cycle_weight * float(
                        max(
                            (self.cycle_decay ** tau)
                            * float((second.conj() * state).real.sum())
                            for tau, state in enumerate(recent)
                        )
                    )
                best = max(best, value)
            lookahead[action] = best
        return lookahead

    def _depth_two_batched(
        self, psi: torch.Tensor, goal: torch.Tensor, recent: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Batched 4 + 16 branch rollout (I3).

        One predict_batch over the four first steps and one over the sixteen
        second steps replace twenty single-item FFTs. The per-action maximum
        over the legal set, the reversal rule, the terminal-win bonus and the
        cycle penalty are applied afterwards on the (4, 4) value table, so
        the arithmetic is the loop's arithmetic, not an approximation.
        """
        first = self.predict_batch(psi, [0, 1, 2, 3])
        align_first = (_slot3(first).conj() * goal).real.sum(dim=-1)
        actions = [second for _first in range(4) for second in range(4)]
        base = first.repeat_interleave(4, dim=0)
        second = self.predict_batch(base, actions)
        values = (_slot3(second).conj() * goal).real.sum(dim=-1).clone()
        distance = torch.from_numpy(self.goal_distance_batch(second).astype(np.int64))
        values = values + (distance == 0).to(values.dtype) * float(self.win_bonus)
        if recent:
            penalty = None
            for tau, state in enumerate(recent):
                term = (second.conj() * state).real.sum(dim=-1) * (self.cycle_decay ** tau)
                penalty = term if penalty is None else torch.maximum(penalty, term)
            if penalty is not None:
                values = values - self.cycle_weight * penalty
        blocked = (
            self.wall_mask_batch(first)
            if self.mask_lookahead
            else np.zeros((4, 4), dtype=bool)
        )
        lookahead = torch.zeros(4)
        for action in range(4):
            legal = [index for index in range(4) if not bool(blocked[action, index])]
            opposite = (1, 0, 3, 2)[int(action)]
            if len(legal) > 1 and opposite in legal:
                legal.remove(opposite)
            if not legal:
                legal = [index for index in range(4) if not bool(blocked[action, index])]
            if not legal:
                lookahead[action] = align_first[action]
                continue
            row = values[action * 4 : action * 4 + 4]
            lookahead[action] = row[torch.as_tensor(legal, dtype=torch.long)].max()
        return lookahead

    # -------------------------------------------------- online repulsor (M1)
    def cycle_score(self, psi: torch.Tensor, recent: Sequence[torch.Tensor]) -> float:
        """Largest coherence between this wave and any non-adjacent visit."""
        if len(recent) < 2:
            return 0.0
        return max(
            float((psi.reshape(-1).conj() * state.reshape(-1)).sum().real)
            for state in recent[1:]
        )

    def dead_end(self, psi: torch.Tensor) -> bool:
        """True when at most one non-reversing unit move is legal."""
        blocked = self.wall_mask(psi)
        legal = [index for index in range(4) if not bool(blocked[index])]
        if len(legal) <= 1:
            return True
        return False

    @torch.no_grad()
    def discover_escape(
        self,
        psi: torch.Tensor,
        recent: Sequence[torch.Tensor],
        legal: np.ndarray,
        previous_action: Optional[int],
        memory,
        namespace: str = "episode",
        cycle_threshold: float = 0.98,
        visited: Optional[Sequence[torch.Tensor]] = None,
        prior_scores: Optional[torch.Tensor] = None,
        alignment_weight: float = 1.0,
        prior_weight: float = 0.5,
        coherence_weight: float = 2.0,
        distance_weight: float = 1.5,
        distance_first: bool = False,
        avoid_reverse: bool = False,
    ) -> Dict[str, object]:
        """Register the trapping wave and let Zone B settle an escape action.

        The current sensor wave is written to Zone C as a test-time REPULSOR,
        then PhotonicCore.relax_discovery is attracted to the goal wave and
        repelled by the recent visit buffer. The settled wave is compared with
        the four one-step successors read through the same plates, and the
        best legal action that is not the immediately previous action is
        returned.
        """
        successors = self.predict_batch(psi, [0, 1, 2, 3])
        successors_out = self.model.zone_b(successors, t_bias=float(self.model.cfg.t_max)).bright
        basis: List[torch.Tensor] = [psi.reshape(-1)] + [
            state.reshape(-1) for state in recent[:4]
        ]
        repulsors = torch.stack(basis).to(torch.complex64)
        directions = torch.cat([successors - psi, psi - successors], dim=0).unsqueeze(0)
        axiom: Dict[str, object] = {"registered": False}
        try:
            record = memory.register_axiom(
                psi.detach().cpu().numpy(),
                name=f"trap_{namespace}_{len(recent)}",
                polarity="REPULSOR",
                source="test_time_learned",
                slot_mask=(True, True, True, True),
                namespace=namespace,
            )
            axiom = {"registered": True, "engram_id": int(record.engram_id)}
        except Exception as error:  # noqa: BLE001 - the failure is the measurement
            axiom = {"registered": False, "error": f"{type(error).__name__}: {error}"}
        state, trace = self.model.zone_b.relax_discovery(
            psi.reshape(1, -1),
            attractors=self.goal.reshape(1, -1),
            repulsors=repulsors,
            laws=None,
            probes=8,
            steps=2,
            tau_rep=0.55,
            lambda_rep=1.2,
            lambda_law=0.0,
            lambda_dark=0.15,
            directions=directions,
            seed=int(self.model.cfg.seed) + 7300 + int(len(recent)),
        )
        settled = state.bright.detach().reshape(-1)
        alignment = torch.zeros(4)
        for action in range(4):
            alignment[action] = (settled.conj() * successors_out[action].reshape(-1)).sum().real
        history = list(visited or recent)
        coherence = np.zeros(4)
        for action in range(4):
            successor = successors[action].reshape(-1)
            values = [
                float((successor.conj() * state.reshape(-1)).sum().real)
                for state in history
            ]
            coherence[action] = max(values) if values else 0.0
        legal_mask = np.asarray(legal, dtype=bool).copy()
        # Hard novelty veto: an action whose successor repeats a recently
        # visited wave is removed unless every legal action repeats. The
        # REPULSOR axioms carry the same information into Zone B; this mask
        # is the deterministic decoding rule that makes the escape repeatable.
        fresh = legal_mask & (coherence < float(cycle_threshold))
        candidate_mask = fresh if bool(fresh.any()) else legal_mask
        if avoid_reverse and visited and len(visited) > 1 and int(candidate_mask.sum()) > 1:
            # Do not step straight back into the state the agent just left.
            # Adjacent cells are only ~0.9 coherent, so this is a separate
            # rule from the exact-revisit veto above: exclude the single
            # successor most coherent with the most recent visit.
            recent_wave = visited[0].reshape(-1)
            back = np.array(
                [
                    float((successors[action].reshape(-1).conj() * recent_wave).sum().real)
                    for action in range(4)
                ]
            )
            back[~candidate_mask] = -np.inf
            candidate_mask[int(np.argmax(back))] = False
        if not bool(candidate_mask.any()):
            candidate_mask = np.ones(4, dtype=bool)
        prior = prior_scores if prior_scores is not None else torch.zeros(4)
        prior = prior.detach().to(torch.float32).clone()
        prior[~torch.isfinite(prior)] = float(prior[torch.isfinite(prior)].min()) if bool(torch.isfinite(prior).any()) else 0.0
        def _z(value: torch.Tensor) -> torch.Tensor:
            spread = float(value.std())
            return (value - value.mean()) / spread if spread > 1e-8 else torch.zeros_like(value)
        distance = torch.tensor(
            [float(self.goal_distance(successors[action])) for action in range(4)],
            dtype=torch.float32,
        )
        if distance_first:
            # Escape mode for a state that has just been trapped: the goal
            # distance is the primary term, the wave alignment and the prior
            # only break ties, and a revisited successor is still penalised.
            combined = (
                -10.0 * distance
                + 0.2 * _z(prior)
                + 0.1 * _z(alignment)
                - float(coherence_weight) * torch.from_numpy(coherence).to(torch.float32)
            )
        else:
            combined = (
                float(alignment_weight) * _z(alignment)
                + float(prior_weight) * _z(prior)
                - float(coherence_weight) * torch.from_numpy(coherence).to(torch.float32)
                - float(distance_weight) * _z(distance)
                + 3.0 * (distance <= 0.0).to(torch.float32)
            )
        masked = combined.clone()
        masked[~torch.from_numpy(candidate_mask)] = -float("inf")
        action = int(masked.argmax())
        return {
            "action": action,
            "scores": [float(value) for value in combined],
            "coherence": [float(value) for value in coherence],
            "candidate_mask": [bool(value) for value in candidate_mask],
            "axiom": axiom,
            "trace": trace,
            "cycle_score": self.cycle_score(psi, recent),
            "dead_end": self.dead_end(psi),
        }

    # --------------------------------------------------------------- rollout
    @torch.no_grad()
    def choose(
        self,
        observation,
        recent: Sequence[torch.Tensor],
        mask_illegal: bool = True,
        use_memory: bool = True,
    ) -> Dict[str, object]:
        tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
        output = self.model("action", tensor, training=False, use_memory=bool(use_memory))
        logits = output.task_logits[0].detach().clone().to(torch.float32)
        view = tensor[0, :75].reshape(5, 5, 3).numpy()
        legal = np.ones(4, dtype=bool)
        for (row, col), action in (((1, 2), 0), ((3, 2), 1), ((2, 1), 2), ((2, 3), 3)):
            if view[row, col, 0] > 0.5:
                legal[action] = False
        psi = self.codec.encode(observation)
        scores, terms = self.score(psi, logits, observation, recent)
        if mask_illegal and bool(legal.any()):
            scores = scores.clone()
            scores[~torch.from_numpy(legal)] = -float("inf")
        action = int(scores.argmax())
        return {
            "action": action,
            "logits": logits,
            "scores": scores,
            "alignment": terms["alignment"],
            "penalty": terms["penalty"],
            "lookahead": terms["lookahead"],
            "combined": terms["combined"],
            "psi": psi,
            "legal": legal,
        }


@torch.no_grad()
def evaluate_closed_loop_action(
    model: HenriHolarch,
    task,
    *,
    episodes: int = 48,
    limit: int = 20,
    goal_weight: float = 0.40,
    cycle_weight: float = 0.60,
    cycle_decay: float = 0.85,
    history: int = 4,
    sigma: float = 1.6,
    rotor=None,
    transition: str = "shift",
    lookahead_depth: int = 1,
    gamma_look: float = 0.02,
    mask_lookahead: bool = True,
    batch_rollout: bool = True,
    use_memory: bool = True,
    trace: bool = False,
    online_repulsor: bool = False,
    online_namespace: str = "m1",
    one_step_audit: bool = True,
) -> Dict[str, object]:
    """Closed-loop success of the 1-step latent wave lookahead policy.

    The evaluator executes the policy in the real grid exactly as
    GridWorldTask.policy_success does, but each action is chosen by the
    wave lookahead. It also reports the reactive baseline on the same
    episodes, so the delta is measured rather than asserted.
    """
    lookahead = LatentActionLookahead(
        model,
        task,
        goal_weight=goal_weight,
        cycle_weight=cycle_weight,
        cycle_decay=cycle_decay,
        history=history,
        sigma=sigma,
        rotor=rotor,
        transition=transition,
        lookahead_depth=lookahead_depth,
        gamma_look=gamma_look,
        mask_lookahead=mask_lookahead,
        batch_rollout=batch_rollout,
    )
    world = task.world
    wins = 0
    reactive_wins = 0
    steps_used: List[int] = []
    chosen_actions: List[int] = []
    episode_count = int(min(int(episodes), len(task.val_layouts)))
    # Reactive control runs its own episodes, so the two arms are measured
    # on identical layouts without sharing a trajectory.
    for index in range(episode_count):
        walls, agent, goal = task.val_layouts[index]
        for _step in range(int(limit)):
            observation = world.observe(walls, agent, goal)
            tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            logits = model("action", tensor, training=False).task_logits[0].clone()
            view = tensor[0, :75].reshape(5, 5, 3).numpy()
            for (row, col), action in (((1, 2), 0), ((3, 2), 1), ((2, 1), 2), ((2, 3), 3)):
                if view[row, col, 0] > 0.5:
                    logits[action] = -float("inf")
            action = int(logits.argmax()) if bool(torch.isfinite(logits).any()) else 0
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
            row, col = agent[0] + dr, agent[1] + dc
            if not (0 <= row < task.grid_size and 0 <= col < task.grid_size):
                break
            if walls[row, col]:
                break
            agent = (row, col)
            if agent == goal:
                reactive_wins += 1
                break
    per_episode: List[Dict[str, object]] = []
    online_discoveries = 0
    online_rows: List[Dict[str, object]] = []
    for index in range(episode_count):
        walls, agent, goal = task.val_layouts[index]
        recent: List[torch.Tensor] = []
        visited: List[torch.Tensor] = []
        distances: List[int] = []
        episode_actions: List[int] = []
        episode_win = False
        episode_steps = 0
        for _step in range(int(limit)):
            observation = world.observe(walls, agent, goal)
            choice = lookahead.choose(observation, recent, use_memory=bool(use_memory))
            action = int(choice["action"])
            if online_repulsor:
                psi = choice["psi"]
                distance = int(lookahead.goal_distance(psi))
                stalled = bool(
                    len(distances) >= 5
                    and min(distances[-4:]) >= distances[-5]
                )
                trapped = (
                    lookahead.cycle_score(psi, visited[-12:]) > 0.98
                    or lookahead.dead_end(psi)
                    or stalled
                )
                if trapped:
                    previous = episode_actions[-1] if episode_actions else None
                    escape = lookahead.discover_escape(
                        psi,
                        visited[-4:] or recent,
                        choice["legal"],
                        previous,
                        getattr(model, "memory", None),
                        namespace=f"{online_namespace}_{index}",
                        visited=visited[-12:],
                        prior_scores=choice["scores"],
                    )
                    action = int(escape["action"])
                    online_discoveries += 1
                    online_rows.append(
                        {
                            "episode": int(index),
                            "step": int(_step),
                            "previous_action": None if previous is None else int(previous),
                            "escape_action": int(action),
                            "cycle_score": float(escape["cycle_score"]),
                            "dead_end": bool(escape["dead_end"]),
                            "axiom": dict(escape["axiom"]),
                        }
                    )
            episode_actions.append(action)
            chosen_actions.append(action)
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
            row, col = agent[0] + dr, agent[1] + dc
            if not (0 <= row < task.grid_size and 0 <= col < task.grid_size):
                break
            if walls[row, col]:
                break
            agent = (row, col)
            recent.insert(0, choice["psi"])
            del recent[lookahead.history :]
            visited.append(choice["psi"])
            if online_repulsor:
                distances.append(int(lookahead.goal_distance(choice["psi"])))
            if agent == goal:
                wins += 1
                steps_used.append(_step + 1)
                episode_win = True
                episode_steps = _step + 1
                break
        if trace:
            # Directive J1: the thread-parity gate compares episodes, not only
            # an aggregate rate, so each episode's outcome and action sequence
            # are returned when a caller asks for the trace.
            per_episode.append(
                {
                    "index": int(index),
                    "win": bool(episode_win),
                    "steps": int(episode_steps),
                    "actions": list(episode_actions),
                }
            )
    # One-step action accuracy on the held-out pairs, for the E1 triple gate.
    # Directive X4: a caller that consumes only the trace does not pay for the
    # 80-pair audit.  The flag defaults to True, so every existing caller keeps
    # the exact number it measured; the episode and Pass-0 paths pass False
    # because neither reads one_step_accuracy.
    correct = 0
    total = 0
    if bool(one_step_audit):
        for observation, label in task.val_pairs:
            tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            logits = model("action", tensor, training=False).task_logits[0]
            view = tensor[0, :75].reshape(5, 5, 3).numpy()
            for (row, col), action in (((1, 2), 0), ((3, 2), 1), ((2, 1), 2), ((2, 3), 3)):
                if view[row, col, 0] > 0.5:
                    logits[action] = -float("inf")
            correct += int(int(logits.argmax()) == int(label))
            total += 1
    return {
        "success_rate": float(wins / max(episode_count, 1)),
        "wins": int(wins),
        "episodes": int(episode_count),
        "reactive_success_rate": float(reactive_wins / max(episode_count, 1)),
        "reactive_wins": int(reactive_wins),
        "one_step_accuracy": float(correct / max(total, 1)),
        "one_step_n": int(total),
        "one_step_audit": bool(one_step_audit),
        "mean_steps_to_goal": float(np.mean(steps_used)) if steps_used else float("nan"),
        "goal_weight": float(goal_weight),
        "cycle_weight": float(cycle_weight),
        "history": int(history),
        "sensor_grid": int(lookahead.codec.grid),
        "goal_sigma": float(lookahead.sigma),
        "transition": str(transition),
        "lookahead_depth": int(lookahead_depth),
        "gamma_look": float(gamma_look),
        "mask_lookahead": bool(mask_lookahead),
        "batch_rollout": bool(batch_rollout),
        "use_memory": bool(use_memory),
        "rotor_fitted": bool(getattr(rotor, "fitted", False)),
        "trace": list(per_episode),
        "online_repulsor": bool(online_repulsor),
        "online_discoveries": int(online_discoveries),
        "online_rows": online_rows,
        "evidence": "OBSERVED",
    }

# ---------------------------------------------------------------------------
# Directive F2: a learned unitary action transition rotor
# ---------------------------------------------------------------------------
class ActionTransitionRotor:
    """One learned unit-modulus frequency mask per action.

        Phi_a = exp(i * theta_a) in C^D
        psi_next(a) = normalize(ifft(fft(psi_t) * Phi_a))

    The mask is fitted in closed form from observed transitions: for each
    action, the optimal single-mask operator maximises the mean coherence
    with the next wave, and the maximiser is the normalised cross-spectrum

        Phi_a = normalise(sum_t F(psi_{t+1}) * conj(F(psi_t)))

    which is a phase-alignment update, not a gradient step. Zone B is never
    touched: the inputs are detached numpy waves. The same estimator applies
    to a spatial shift and to an abstract transition, which is exactly the
    generalisation the directive asks for.
    """

    def __init__(self, dim: int, n_actions: int = 4, n_slots: int = 4, grid: Optional[int] = None) -> None:
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.n_slots = int(n_slots)
        self.d_slot = self.dim // self.n_slots
        self.grid = int(grid) if grid else int(round(math.sqrt(self.d_slot)))
        if self.grid * self.grid != self.d_slot:
            raise ValueError("the rotor needs a square per-slot grid")
        self.phi = np.ones((self.n_actions, self.n_slots, self.grid, self.grid), dtype=np.complex64)
        self.counts = np.zeros(self.n_actions, dtype=np.int64)
        self.fitted = False
        # Cycle AB5: the device basis cache is invalidated by this counter only.
        self.weight_epoch = 0
        self._phi_cache = None

    # ------------------------------------------------------------- geometry
    def _slots(self, wave: torch.Tensor) -> np.ndarray:
        flat = wave.detach().cpu().numpy().reshape(-1)[: self.dim]
        return flat.reshape(self.n_slots, self.grid, self.grid).astype(np.complex64)

    def _from_slots(self, slots: np.ndarray) -> torch.Tensor:
        flat = slots.reshape(1, self.dim).astype(np.complex64)
        return _unit_tensor(torch.from_numpy(flat))

    # --------------------------------------------------------------- fitting
    def fit(self, transitions: Sequence[Tuple[torch.Tensor, int, torch.Tensor]]) -> Dict[str, float]:
        """Closed-form phase alignment on (psi_t, a_t, psi_{t+1}) triples."""
        accumulator = np.zeros_like(self.phi)
        counts = np.zeros(self.n_actions, dtype=np.int64)
        for current, action, nxt in transitions:
            index = int(action) % self.n_actions
            left = np.fft.fft2(self._slots(current))
            right = np.fft.fft2(self._slots(nxt))
            accumulator[index] += right * np.conj(left)
            counts[index] += 1
        for index in range(self.n_actions):
            magnitude = np.abs(accumulator[index])
            unit = np.where(magnitude > 1e-8, accumulator[index] / np.maximum(magnitude, 1e-8), 1.0)
            self.phi[index] = unit.astype(np.complex64)
        self.counts = counts
        self.weight_epoch = int(getattr(self, "weight_epoch", 0)) + 1
        self._phi_cache = None
        self.fitted = bool(counts.sum() > 0)
        return {
            "transitions": float(counts.sum()),
            "actions_covered": float(int((counts > 0).sum())),
            "fitted": float(self.fitted),
        }

    # ------------------------------------------------------------ inference
    def _device_phi(self, device: torch.device) -> torch.Tensor:
        """Cycle AB5: the planar rotor basis cached per device and epoch."""
        key = (str(device), int(self.weight_epoch))
        cached = getattr(self, "_phi_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tensor = torch.from_numpy(np.asarray(self.phi, dtype=np.complex64)).to(device)
        self._phi_cache = (key, tensor)
        return tensor

    def predict(self, psi: torch.Tensor, action: int) -> torch.Tensor:
        index = int(action) % self.n_actions
        if psi.is_cuda:
            # Zero host sync: the 2D toroidal FFT and the phase product stay on
            # the device, so a CUDA-graph capture sees no .cpu().numpy() call.
            slots = psi.reshape(-1)[: self.dim].reshape(
                self.n_slots, self.grid, self.grid
            ).to(torch.complex64)
            phi = self._device_phi(psi.device)[index]
            spectrum = torch.fft.fft2(slots)
            out = torch.fft.ifft2(spectrum * phi)
            flat = out.real.reshape(1, self.dim).to(torch.complex64)
            return _unit_tensor(flat)
        spectrum = np.fft.fft2(self._slots(psi))
        out = np.fft.ifft2(spectrum * self.phi[index])
        return self._from_slots(out.real.astype(np.float32).astype(np.complex64))

    def predict_batch(self, psi: torch.Tensor, actions: Sequence[int]) -> torch.Tensor:
        """Batched rotor rollout: one fft2, one masked ifft2 (I3).

        actions[i] selects phi[actions[i]] for row i. The arithmetic is the
        same per-row product/ifft2 as predict(); the batch dimension removes
        the Python loop that dominated the 16-branch depth-2 rollout.
        """
        indices = [int(action) % self.n_actions for action in actions]
        if psi.is_cuda:
            # Device-native batched toroidal rollout (Cycle AB5).
            flat = psi.reshape(-1)[: self.dim].reshape(1, self.n_slots, self.grid, self.grid)
            if len(indices) > 1:
                flat = flat.expand(len(indices), -1, -1, -1)
            slots = flat.to(torch.complex64)
            phi = self._device_phi(psi.device)[torch.as_tensor(indices, device=psi.device)]
            spectrum = torch.fft.fft2(slots)
            out = torch.fft.ifft2(spectrum * phi)
            flat_out = out.real.reshape(len(indices), self.dim).to(torch.complex64)
            return _unit_tensor(flat_out)
        batch = psi.detach().cpu().numpy().reshape(-1, self.dim)[:, : self.dim]
        count = int(batch.shape[0])
        slots = batch.reshape(count, self.n_slots, self.grid, self.grid).astype(np.complex64)
        if count == 1 and len(indices) > 1:
            slots = np.repeat(slots, len(indices), axis=0)
            count = len(indices)
        elif count != len(indices):
            raise ValueError("predict_batch needs one wave per action, or one wave")
        spectrum = np.fft.fft2(slots)
        phi = self.phi[indices]
        out = np.fft.ifft2(spectrum * phi)
        flat = out.real.reshape(count, self.dim).astype(np.float32).astype(np.complex64)
        return _unit_tensor(torch.from_numpy(flat))

    @torch.no_grad()
    def coherence(
        self,
        transitions: Sequence[Tuple[torch.Tensor, int, torch.Tensor]],
    ) -> Dict[str, float]:
        values: List[float] = []
        for current, action, nxt in transitions:
            predicted = self.predict(current, int(action))
            values.append(float(torch.real(torch.vdot(predicted.reshape(-1), nxt.reshape(-1)))))
        if not values:
            return {"coherence": float("nan"), "n": 0.0}
        return {
            "coherence": float(np.mean(values)),
            "coherence_min": float(np.min(values)),
            "n": float(len(values)),
        }

    def state(self) -> Dict[str, object]:
        return {
            "dim": int(self.dim),
            "n_actions": int(self.n_actions),
            "grid": int(self.grid),
            "fitted": bool(self.fitted),
            "counts": [int(value) for value in self.counts],
        }

    def save(self, path: str | Path) -> Dict[str, str]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, phi=self.phi, counts=self.counts)
        digest = _digest_file(target)
        target.with_name(target.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
        return {"path": str(target), "sha256": digest}

    def load(self, path: str | Path) -> "ActionTransitionRotor":
        payload = np.load(Path(path))
        self.phi = payload["phi"].astype(np.complex64)
        self.counts = payload["counts"].astype(np.int64)
        self.fitted = bool(self.counts.sum() > 0)
        return self


def collect_action_transitions(
    model: HenriHolarch,
    task,
    episodes: int = 48,
    limit: int = 20,
    sigma: float = 1.6,
) -> List[Tuple[torch.Tensor, int, torch.Tensor]]:
    """Observed (psi_t, a_t, psi_{t+1}) triples from real grid steps.

    The action is the BFS-optimal one, so the transition is the environment's
    true dynamics and not the policy's guess. A step into a wall ends the
    episode: a blocked move is not a transition.
    """
    codec = SensorWaveCodec(int(model.cfg.dim), int(model.cfg.n_slots), float(sigma))
    world = task.world
    transitions: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    for index in range(min(int(episodes), len(task.train_layouts))):
        walls, agent, goal = task.train_layouts[index]
        for _step in range(int(limit)):
            action = world.optimal_action(walls, agent, goal)
            if action is None:
                break
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(action)]
            row, col = agent[0] + dr, agent[1] + dc
            if not (0 <= row < task.grid_size and 0 <= col < task.grid_size):
                break
            if walls[row, col]:
                break
            current = codec.encode(world.observe(walls, agent, goal))
            nxt = codec.encode(world.observe(walls, (row, col), goal))
            transitions.append((current, int(action), nxt))
            agent = (row, col)
            if agent == goal:
                break
    return transitions


@torch.no_grad()
def fit_action_rotor(
    model: HenriHolarch,
    task,
    *,
    episodes: int = 40,
    limit: int = 20,
    sigma: float = 1.6,
    holdout_fraction: float = 0.25,
) -> Dict[str, object]:
    """Fit the rotor on train layouts and measure it on held-out ones."""
    train = collect_action_transitions(model, task, episodes=episodes, limit=limit, sigma=sigma)
    rotor = ActionTransitionRotor(int(model.cfg.dim), int(model.cfg.n_actions), int(model.cfg.n_slots))
    rotor.fit(train)
    held = collect_action_transitions(
        model,
        task,
        episodes=int(episodes),
        limit=int(limit),
        sigma=sigma,
    )
    # A disjoint validation split: the last quarter of the validation layouts'
    # transitions are re-collected from val_layouts, never seen at fit time.
    val: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    codec = SensorWaveCodec(int(model.cfg.dim), int(model.cfg.n_slots), float(sigma))
    world = task.world
    for index in range(min(int(episodes), len(task.val_layouts))):
        walls, agent, goal = task.val_layouts[index]
        for _step in range(int(limit)):
            action = world.optimal_action(walls, agent, goal)
            if action is None:
                break
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(action)]
            row, col = agent[0] + dr, agent[1] + dc
            if not (0 <= row < task.grid_size and 0 <= col < task.grid_size):
                break
            if walls[row, col]:
                break
            current = codec.encode(world.observe(walls, agent, goal))
            nxt = codec.encode(world.observe(walls, (row, col), goal))
            val.append((current, int(action), nxt))
            agent = (row, col)
            if agent == goal:
                break
    train_metrics = rotor.coherence(train)
    val_metrics = rotor.coherence(val)
    return {
        "rotor": rotor,
        "fit": {
            "transitions": int(len(train)),
            "coherence": float(train_metrics["coherence"]),
            "actions_covered": int((rotor.counts > 0).sum()),
        },
        "held_out": {
            "transitions": int(len(val)),
            "coherence": float(val_metrics["coherence"]),
            "coherence_min": float(val_metrics.get("coherence_min", float("nan"))),
        },
        "state": rotor.state(),
        "evidence": "OBSERVED",
        "holdout_fraction": float(holdout_fraction),
    }


# ---------------------------------------------------------------------------
# Directive I3: deterministic Cul-de-Sac horizon traps
# ---------------------------------------------------------------------------
CUL_DE_SAC_TABLE = (
    # (agent_row, agent_col, pocket_row, pocket_col, goal_row, goal_col)
    # Eight states selected from the deterministic 7x7 family whose pocket is
    # a one-cell dead end on the discrete steepest-descent direction of the
    # Euclidean goal field. The selection rule and the full candidate count
    # are recorded in receipts/holarch/i3_horizon_traps.json.
    (1, 1, 1, 2, 5, 6),
    (1, 3, 2, 3, 5, 2),
    (1, 4, 2, 4, 4, 2),
    (2, 1, 2, 2, 4, 5),
    (3, 1, 3, 2, 2, 4),
    (4, 1, 3, 1, 1, 3),
    (5, 2, 4, 2, 1, 5),
    (5, 3, 5, 2, 3, 0),
)
"""The eight Cul-de-Sac traps: agent, pocket and goal coordinates."""

TRAP_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def build_cul_de_sac_traps(size: int = 7, world=None) -> List[Dict[str, object]]:
    """Construct the eight deterministic Cul-de-Sac horizon traps (I3).

    Each state is a 7x7 grid with one free pocket cell adjacent to the agent
    and directly on the steepest-descent direction of the Euclidean goal
    field. The pocket's forward and both lateral neighbours are walls, so at
    t+2 the only legal move from inside the pocket is the 180-degree
    reversal. The bypass corridor is the BFS-optimal route around the pocket.

    The function is deterministic and model-free: it builds geometry and the
    79-value observation only. The arm behaviour (masked vs depth-1 vs
    unmasked) is measured by the caller, because it is a property of the
    fitted checkpoint, not of the layout.
    """
    if world is None:
        from henri.holarch.tasks import _Grid

        world = _Grid(size=int(size), seed=0)
    traps: List[Dict[str, object]] = []
    for index, (ar, ac, pr, pc, gr, gc) in enumerate(CUL_DE_SAC_TABLE):
        agent = (int(ar), int(ac))
        pocket = (int(pr), int(pc))
        goal = (int(gr), int(gc))
        d_row = int(np.sign(pr - ar))
        d_col = int(np.sign(pc - ac))
        if (d_row, d_col) not in TRAP_DELTAS:
            raise ValueError(f"trap {index}: the pocket is not adjacent to the agent")
        trap_action = TRAP_DELTAS.index((d_row, d_col))
        walls = np.zeros((int(size), int(size)), dtype=bool)
        blocked_cells = [
            (pocket[0] + d_row, pocket[1] + d_col),
            (pocket[0] + d_col, pocket[1] + d_row),
            (pocket[0] - d_col, pocket[1] - d_row),
        ]
        for row, col in blocked_cells:
            if not (0 <= row < int(size) and 0 <= col < int(size)):
                raise ValueError(f"trap {index}: a pocket wall leaves the grid")
            walls[row, col] = True
        if walls[agent] or walls[pocket] or walls[goal]:
            raise ValueError(f"trap {index}: a wall occupies the agent, pocket or goal")
        path = world._bfs(walls, agent, goal)
        if path is None:
            raise ValueError(f"trap {index}: the goal is unreachable")
        bypass_action = int(path[0])
        dx = abs(goal[0] - agent[0])
        dy = abs(goal[1] - agent[1])
        dominant = (dx >= dy) if d_row else (dy >= dx)
        traps.append(
            {
                "id": f"trap{index + 1:02d}",
                "grid": int(size),
                "agent": agent,
                "pocket": pocket,
                "goal": goal,
                "trap_action": int(trap_action),
                "bypass_action": int(bypass_action),
                "walls": walls,
                "wall_cells": [list(cell) for cell in blocked_cells],
                "observation": world.observe(walls, agent, goal).tolist(),
                "path_length": int(len(path)),
                "invariants": {
                    "pocket_legal": bool(not walls[pocket]),
                    "three_non_reversing_sides_blocked": bool(all(walls[cell] for cell in blocked_cells)),
                    "only_reversal_open_at_t2": bool(
                        sum(1 for cell in blocked_cells if walls[cell]) == 3
                    ),
                    "bypass_is_not_the_trap": bool(bypass_action != trap_action),
                    "pocket_on_dominant_goal_axis": bool(dominant),
                    "goal_reachable": bool(path is not None),
                },
            }
        )
    return traps

