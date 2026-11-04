"""HENRI holarch configuration.

Evidence label: DERIVED. Every constant is a declared design choice or a
budget bound, not a measured capability.

The holarch is the three-zone system:

    Zone A  lean continuous-learning swarm (ingress inference and action)
    Zone B  BaTiO3 D2NN + Sagnac photonic kernel (wave reasoning)
    Zone C  holographic TimescaleDB/NVMe agentic memory (context)

One immutable config fixes every dimension so a receipt can be replayed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

VRAM_BUDGET_GB: float = 32.0
"""The RTX 5090 / Blackwell consumer budget this build must fit inside."""

DEFAULT_DIM: int = 2048
"""Complex wave dimension. Divisible by 4 slots and by 8 Cl(3,0) blocks."""

PRODUCTION_PLATES: int = 4
"""BaTiO3 Kerr plates in the production stack (Decision B4).

The four-plate cascade is the configuration that closed Gate D1 strict at
+0.111 mean plate margin (+0.082 on 8x8, +0.140 on 16x16 clutter). Three
plates measured +0.098 and two measured +0.081, so the depth is a measured
requirement of the strict gate, not a preference. The CI profile keeps two
plates for speed; every profile that asserts Gate D1 strict pins this value.
"""


def _default_device() -> str:
    """Return 'cuda' when a device is present, else 'cpu'. Never guesses."""
    override = os.environ.get("HENRI_DEVICE", "").strip().lower()
    if override in {"cpu", "cuda"}:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch is a hard dependency of the holarch
        return "cpu"


@dataclass(frozen=True)
class HolarchConfig:
    """Immutable dimensions, physics constants, and budgets.

    Args:
        dim: complex wave dimension. Must be divisible by
            n_slots * 8 so the four slots and the Clifford blocks tile
            the vector exactly (constraint C6).
        n_slots: structural slots. Fixed at four by the architecture.
        photonic_layers: cascaded BTO diffractive plates in Zone B.
        pockels_gain: phase authority per unit normalised voltage.
        kerr: base Kerr self-phase coefficient (learned as softplus).
        curie_kappa: Curie-Weiss susceptibility weight.
        curie_eps: regulariser that keeps the gain finite at T -> Tc.
        t_max: maximum normalised temperature. Below 1 so the gain is
            finite and the device stays in the paraelectric phase.
        propagation_zeta: 4f propagation strength in radians at the
            Nyquist frequency.
        voltage_bound: hard bound on the normalised drive voltage.
        vision_size: canvas side of the vision sensor. 8 is the sklearn
            8x8 grid (d_input = 64). 16 is the 2x2 patch-composition
            canvas (d_input = 256) used by the Gate D1 bottleneck probe.
        vision_composition: label rule of the 16x16 canvas. 'sum' is
            (d0 + d1) mod 10 over the two informative quadrants, 'diff' is
            |d0 - d1|. 'clutter' labels the canvas by quadrant 0 and fills
            the other three quadrants with independent distractor digits,
            so 75 percent of the input energy is task irrelevant.
        vision_encoder: 'conv' learned convolutional encoder, or 'slm' a
            parameter-free phase-only spatial light modulator. 'slm'
            removes the learned encoder so the only nonlinearity between
            pixels and the readout is the photonic stack.
        slm_gain: phase radians per unit of standardised pixel value for
            the phase-only sensor.
        readout: 'swarm' (Zone A experts) or 'linear' (one linear probe
            over the pooled photonic features). 'linear' is the Gate D1
            attribution probe: it removes the Zone A MLP so any accuracy
            difference is attributable to the optics.
        d_feat: feature width handed to Zone A.
        n_experts: lean experts in the Zone A swarm.
        expert_hidden: per-expert hidden width.
        fast_rank: rank of each expert's fast-weight associative memory.
        top_k: experts mixed per token.
        n_tasks: task heads.
        vocab_size: character vocabulary for the language task.
        n_vision_classes: digit classes.
        n_actions: grid-world action count.
        relational_binding: 'wave' binds the three role phasors in the
            d_slot complex domain and reads them through a dedicated
            Stiefel basis (Decision 3), 'trilinear' is the retired
            K-dim real product (Decision C control arm).
        relational_k: complex rows of the relational Stiefel basis. Both
            quadratures are read, so the head width is 2 * relational_k per
            bound wave.
        relational_shifts: number of cyclic frequency shifts bound by
            circular convolution (Decision 7). One shift reproduces the
            single diagonal product; four cross-mix frequency channels.
        relational_head: 'expert' routes the bound block through the Zone A
            relational MLP (Decision 7); 'linear' keeps the old standalone
            linear head.
        universal_rank: rank of the shared Zone A weight subspace that all
            four tasks read through (Decision 8). 0 disables it.
        relational_hidden: width of the dedicated relational expert. The CI
            swarm expert is 48 wide; Decision 7 asks the relational path to
            match the MLP baseline's non-linear capacity, so it gets its own
            two-layer micro-net over [g0||g1||g2||g_rel].
        relational_dropout: dropout on the relational feature block.
        relational_l2: weight decay applied to the relational head only.
        relax_mode: 'prototype' anchors relaxation in the clean
            post-photonic prototype bank (Decision 2), 'axiom' is the
            retired pull toward the post-photonic context bundle.
        prototype_lambda_dark: weight of the Stiefel nullspace term in the
            relaxation energy.
        prototype_eta: egress phase step per relaxation pass.
        prototype_beta: inverse temperature of the prototype competition.
            0 means calibrate it from the bank (constraint C3 refuses the
            transferred literal).
        relax_training: run the relaxation loop during training as well, so
            the readout is trained on the cleaned wave it will be served.
            Off by default: every pass costs one extra propagation.
        prototype_logits: add the calibrated Hopfield prototype logits
            beta * Re<psi, phi_c> to the task logits (Decision 6). The head
            lets the relaxation loop change the DECISION, not only the wave.
        proto_alpha_init: initial scale of the prototype logit block.
        prototype_trust_floor: eta at zero surprise (Decision A2). The
            prototype branch keeps at least this weight, so a familiar
            input still benefits from the attractor readout.
        prototype_trust_mid: surprise at which eta is halfway between the
            floor and one. Recalibrated per task from the clean surprise
            distribution when the gate is calibrated.
        prototype_trust_scale: logistic width of the trust ramp.
        stiefel_mode: 'learned' keeps the Gate D1 gauge freedom (a free
            Stiefel basis absorbs any unitary optical transform). 'fixed'
            anchors the rows to a canonical Fourier detector frame, so the
            plates are the only beam-steerer (Decision 5).
        kerr_interplate: electro-optic intensity coefficient. The plate
            phase becomes theta + dir * g * Lambda(T) * (V + gamma |psi|^2),
            which makes the stack non-linear in amplitude and impossible to
            collapse into one matrix multiply (Decision 5).
        fast_gate: gate Zone A fast-weight plasticity on Zone B surprise.
        fast_gate_tau: surprise threshold on the prototype or Sagnac
            residual above which plasticity opens.
        fast_gate_familiar: raw Gram coherence below which plasticity
            opens even when the residual is small.
        gram_dims: complex projection width stored per Zone C engram.
        chart_capacity: engrams per holographic chart before compaction.
        cache_capacity: resident wave cache (bounded, paged from disk).
        context_tokens: fixed context tensor rows (invariant in N).
        context_dim: fixed context tensor columns.
        tau_seconds: viscoelastic relaxation timescales.
        tau_weights: per-timescale weights.
        crosstalk_max: Gram crosstalk above which compaction runs.
        prune_weight: viscoelastic weight below which working engrams drop.
        device: 'cpu', 'cuda', or 'auto'.
        seed: explicit seed for every stochastic path.
    """

    dim: int = DEFAULT_DIM
    n_slots: int = 4

    photonic_layers: int = PRODUCTION_PLATES
    pool_k: int = 16
    cross_slot_mix: bool = True
    skip_alpha: float = 0.0
    relax_steps: int = 4
    alpha_thermal: float = 0.35
    relax_fraction: float = 0.5
    pockels_gain: float = 0.9
    kerr: float = 0.08
    curie_kappa: float = 0.9
    curie_eps: float = 1e-3
    t_max: float = 0.85
    propagation_zeta: float = 1.0
    voltage_bound: float = 1.0

    vision_size: int = 8
    vision_composition: str = "sum"
    vision_encoder: str = "conv"
    slm_gain: float = 1.0
    readout: str = "swarm"

    relax_mode: str = "prototype"
    prototype_lambda_dark: float = 0.5
    prototype_eta: float = 0.15
    prototype_beta: float = 0.0
    relax_training: bool = False
    prototype_logits: bool = True
    proto_alpha_init: float = 0.05
    proto_relax_gain: float = 1.0
    """Decision B2 gain on the surprise-gated relaxation logit delta.

    The Pass-0 prototype logit block is always added at alpha. This gain
    multiplies only (proto(relaxed) - proto(pass0)), so the clean input is
    untouched when the loop does not move the wave.
    """
    prototype_trust_floor: float = 0.35
    prototype_trust_mid: float = 0.30
    prototype_trust_scale: float = 0.08

    stiefel_mode: str = "learned"
    kerr_interplate: float = 0.0

    fast_gate: bool = False
    fast_gate_tau: float = 0.05
    fast_gate_familiar: float = 0.0

    d_latent: int = 128
    d_feat: int = 192
    n_experts: int = 8
    expert_hidden: int = 192
    fast_rank: int = 32
    top_k: int = 2
    n_tasks: int = 4
    vocab_size: int = 96
    n_vision_classes: int = 10
    n_actions: int = 4
    obs_dim: int = 79
    relational_entities: int = 6
    relational_actions: int = 6
    relational_targets: int = 8
    relational_classes: int = 8
    relational_binding: str = "conv_shift"
    relational_k: int = 16
    relational_shifts: int = 4
    relational_head: str = "expert"
    relational_dropout: float = 0.05
    relational_l2: float = 5e-4
    relational_hidden: int = 192
    universal_rank: int = 8

    gram_dims: int = 64
    chart_capacity: int = 256
    cache_capacity: int = 256
    context_tokens: int = 16
    context_dim: int = 64
    tau_seconds: tuple = (1.0, 60.0, 3600.0, float("inf"))
    tau_weights: tuple = (0.4, 0.3, 0.2, 0.1)
    crosstalk_max: float = 0.12
    prune_weight: float = 1e-4
    consolidation_tier: int = 3

    device: str = "auto"
    seed: int = 20261102

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        span = self.n_slots * 8
        if self.dim % span != 0:
            raise ValueError(f"dim={self.dim} must be divisible by n_slots*8={span}")
        if self.n_slots != 4:
            raise ValueError("the 4-slot structural rule is frozen at n_slots == 4")
        if self.dim % self.n_slots != 0:
            raise ValueError("dim must be divisible by n_slots")
        if self.photonic_layers < 1:
            raise ValueError("photonic_layers must be >= 1")
        if self.pool_k < 1 or self.pool_k > self.d_slot:
            raise ValueError("pool_k must lie in [1, d_slot]")
        if not 0.0 <= self.skip_alpha <= 1.0:
            raise ValueError("skip_alpha must lie in [0, 1]")
        if self.relax_steps < 0:
            raise ValueError("relax_steps must be >= 0")
        if not 0.0 <= self.alpha_thermal <= 1.0:
            raise ValueError("alpha_thermal must lie in [0, 1]")
        if not 0.0 <= self.relax_fraction <= 1.0:
            raise ValueError("relax_fraction must lie in [0, 1]")
        if not 0.0 < self.t_max < 1.0:
            raise ValueError("t_max must lie in (0, 1) so the Curie gain is finite")
        if not 0.0 < self.curie_eps < 1.0:
            raise ValueError("curie_eps must lie in (0, 1)")
        if self.pockels_gain <= 0.0:
            raise ValueError("pockels_gain must be positive")
        if self.kerr <= 0.0:
            raise ValueError("kerr must be positive")
        if self.voltage_bound <= 0.0:
            raise ValueError("voltage_bound must be positive")
        if self.vision_size not in {8, 16}:
            raise ValueError("vision_size must be 8 or 16")
        if self.vision_composition not in {"sum", "diff", "clutter"}:
            raise ValueError("vision_composition must be 'sum', 'diff', or 'clutter'")
        if self.vision_encoder not in {"conv", "slm"}:
            raise ValueError("vision_encoder must be 'conv' or 'slm'")
        if self.vision_size ** 2 > self.dim:
            raise ValueError("vision_size**2 must not exceed dim")
        if self.slm_gain <= 0.0:
            raise ValueError("slm_gain must be positive")
        if self.readout not in {"swarm", "linear"}:
            raise ValueError("readout must be 'swarm' or 'linear'")
        if self.relax_mode not in {"prototype", "axiom"}:
            raise ValueError("relax_mode must be 'prototype' or 'axiom'")
        if self.prototype_lambda_dark < 0.0:
            raise ValueError("prototype_lambda_dark must be >= 0")
        if not 0.0 <= self.prototype_eta <= 1.0:
            raise ValueError("prototype_eta must lie in [0, 1]")
        if self.prototype_beta < 0.0:
            raise ValueError("prototype_beta must be >= 0")
        if self.relational_binding not in {"conv_shift", "wave", "trilinear"}:
            raise ValueError("relational_binding must be 'conv_shift', 'wave', or 'trilinear'")
        if self.relational_head not in {"expert", "linear"}:
            raise ValueError("relational_head must be 'expert' or 'linear'")
        if not 1 <= self.relational_k <= self.d_slot:
            raise ValueError("relational_k must lie in [1, d_slot]")
        if not 1 <= self.relational_shifts <= self.d_slot:
            raise ValueError("relational_shifts must lie in [1, d_slot]")
        if not 0 <= self.universal_rank <= self.d_feat:
            raise ValueError("universal_rank must lie in [0, d_feat]")
        if self.stiefel_mode not in {"learned", "fixed"}:
            raise ValueError("stiefel_mode must be 'learned' or 'fixed'")
        if self.kerr_interplate < 0.0:
            raise ValueError("kerr_interplate must be >= 0")
        if self.proto_alpha_init < 0.0:
            raise ValueError("proto_alpha_init must be >= 0")
        if not 0.0 <= self.prototype_trust_floor <= 1.0:
            raise ValueError("prototype_trust_floor must lie in [0, 1]")
        if self.prototype_trust_scale <= 0.0:
            raise ValueError("prototype_trust_scale must be positive")
        if not 0.0 <= self.relational_dropout < 1.0:
            raise ValueError("relational_dropout must lie in [0, 1)")
        if self.relational_l2 < 0.0:
            raise ValueError("relational_l2 must be >= 0")
        if self.n_experts < 1 or self.expert_hidden < 1:
            raise ValueError("swarm widths must be positive")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must lie in [1, n_experts]")
        if self.fast_rank < 1:
            raise ValueError("fast_rank must be >= 1")
        if self.gram_dims < 1 or self.gram_dims > self.dim:
            raise ValueError("gram_dims must lie in [1, dim]")
        if self.context_tokens < 1 or self.context_dim < 1:
            raise ValueError("context tensor dimensions must be positive")
        if self.dim % self.context_tokens != 0:
            raise ValueError("dim must be divisible by context_tokens")
        if len(self.tau_seconds) != len(self.tau_weights):
            raise ValueError("tau_seconds and tau_weights must align")
        if not 0.0 < self.crosstalk_max < 1.0:
            raise ValueError("crosstalk_max must lie in (0, 1)")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

    # -- derived views -------------------------------------------------

    @property
    def d_slot(self) -> int:
        """Complex components per structural slot."""
        return self.dim // self.n_slots

    @property
    def n_blocks(self) -> int:
        """Number of Cl(3,0) multivector blocks."""
        return self.dim // 8

    @property
    def chunk(self) -> int:
        """Components per context-tensor row."""
        return self.dim // self.context_tokens

    def slot_slice(self, index: int) -> slice:
        if not 0 <= index < self.n_slots:
            raise IndexError(f"slot {index} out of range")
        return slice(index * self.d_slot, (index + 1) * self.d_slot)

    def resolved_device(self) -> str:
        return _default_device() if self.device == "auto" else self.device

    def replace(self, **changes) -> "HolarchConfig":
        return replace(self, **changes)

    def digest(self) -> str:
        """Stable short hash of the whole configuration."""
        import hashlib
        import json

        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    # -- constructors --------------------------------------------------

    @classmethod
    def small(cls, **overrides) -> "HolarchConfig":
        """Fast CI configuration with the same structure and physics."""
        base = dict(
            dim=256,
            photonic_layers=2,
            d_latent=32,
            d_feat=48,
            pool_k=16,
            n_experts=4,
            expert_hidden=48,
            fast_rank=8,
            top_k=2,
            gram_dims=16,
            chart_capacity=32,
            cache_capacity=16,
            context_tokens=8,
            context_dim=16,
            device="cpu",
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def production(cls, **overrides) -> "HolarchConfig":
        """Full-size configuration for the 32 GiB Blackwell target."""
        base = dict(dim=4096, photonic_layers=PRODUCTION_PLATES, d_feat=256, n_experts=16,
                    expert_hidden=256, fast_rank=64, pool_k=32)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def blackwell(cls, **overrides) -> "HolarchConfig":
        """Scaled profile that still fits 32 GiB with wide headroom.

        This profile is budgeted and constructible. It is not trained in
        this checkout: no CUDA device is present, and a CPU run at this
        width would be slow enough to invite a misleading cost claim.
        """
        base = dict(
            dim=8192,
            photonic_layers=6,
            d_latent=256,
            d_feat=512,
            n_experts=32,
            expert_hidden=512,
            fast_rank=128,
            gram_dims=128,
            pool_k=64,
        )
        base.update(overrides)
        return cls(**base)


TASK_NAMES = ("vision", "language", "action", "relational")
TASK_IDS = {name: index for index, name in enumerate(TASK_NAMES)}
