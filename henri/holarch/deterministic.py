"""Canonical, thread-invariant tensor contractions (Directive J1).

Measured boundary. The holarch's structural contractions started as complex
torch.einsum expressions. On this host, with the same input and the same
process, forty repetitions of the optical slot readout produced up to nine
distinct float results at torch.get_num_threads() == 8 and exactly one at one
thread. The kernel under that path is torch.bmm: eighty repetitions of the same
(4, 1, 64) x (4, 64, 16) complex product differed by up to 1.6e-1 in absolute
value, which is far larger than any rounding argument can explain. A one- or
two-dimensional torch.matmul over the same operands was bit-identical on every
repetition.

So the rule this module enforces is narrower than "use float64": never call a
batched matrix-multiply kernel for a structural contraction. Each helper below
loops over the (four) slots and issues a two-dimensional product, splitting the
complex operands into real and imaginary float64 matrices so the accumulation
order is fixed by the layout and not by a thread schedule. The result is cast
back to complex64, the dtype the rest of the zone works in.

The helpers are pure functions: no state, no lock, and no mutation of their
arguments.
"""

from __future__ import annotations

from typing import Optional

import torch

ACCUMULATE = torch.float64
"""Accumulator dtype for every canonical contraction."""


def _mm(left: torch.Tensor, right: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    """2-D complex product with a real-float64 accumulator.

    Args:
        left: complex tensor of shape (n, m).
        right: complex tensor of shape (m, p).
    """
    lr = left.real.to(ACCUMULATE)
    li = left.imag.to(ACCUMULATE)
    rr = right.real.to(ACCUMULATE)
    ri = right.imag.to(ACCUMULATE)
    real = lr @ rr - li @ ri
    imag = lr @ ri + li @ rr
    target = torch.complex64 if dtype is None else dtype
    return torch.complex(real, imag).to(target)


def stiefel_readout(
    slots: torch.Tensor,
    basis: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """g[b, s, k] = sum_d slots[b, s, d] * conj(basis[s, k, d]).

    Args:
        slots: complex tensor of shape (batch, n_slots, d_slot).
        basis: complex tensor of shape (n_slots, k, d_slot).

    Returns:
        complex tensor of shape (batch, n_slots, k).
    """
    batch, n_slots = int(slots.shape[0]), int(slots.shape[1])
    k = int(basis.shape[1])
    target = torch.complex64 if dtype is None else dtype
    out = torch.zeros(batch, n_slots, k, dtype=torch.complex128)
    for index in range(n_slots):
        out[:, index, :] = out[:, index, :] + _mm(
            slots[:, index, :], basis[index].conj().transpose(-1, -2), torch.complex128
        )
    return out.to(target)


def stiefel_reproject(
    inner: torch.Tensor,
    basis: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """back[b, s, d] = sum_k inner[b, s, k] * basis[s, k, d]."""
    batch, n_slots = int(inner.shape[0]), int(inner.shape[1])
    width = int(basis.shape[-1])
    target = torch.complex64 if dtype is None else dtype
    out = torch.zeros(batch, n_slots, width, dtype=torch.complex128)
    for index in range(n_slots):
        out[:, index, :] = out[:, index, :] + _mm(
            inner[:, index, :], basis[index], torch.complex128
        )
    return out.to(target)


def slot_mix(
    matrix: torch.Tensor,
    slots: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """out[b, i, d] = sum_j matrix[i, j] * slots[b, j, d].

    The slot count is four, so the explicit sum is 16 multiply-adds of a
    (batch, d_slot) row. Elementwise products are not a parallel reduction, so
    the arithmetic is order-free.
    """
    n_slots = int(slots.shape[1])
    target = torch.complex64 if dtype is None else dtype
    rows = []
    for row in range(n_slots):
        accumulator = torch.zeros_like(slots[:, 0, :], dtype=torch.complex128)
        for column in range(n_slots):
            gain = matrix[row, column].to(torch.complex128)
            accumulator = accumulator + gain * slots[:, column, :].to(torch.complex128)
        rows.append(accumulator.to(target))
    return torch.stack(rows, dim=1)


def prototype_align(
    slots: torch.Tensor,
    phi_slots: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """out[b, s, m] = sum_d conj(slots[b, s, d]) * phi_slots[m, s, d]."""
    batch, n_slots = int(slots.shape[0]), int(slots.shape[1])
    count = int(phi_slots.shape[0])
    target = torch.complex64 if dtype is None else dtype
    out = torch.zeros(batch, n_slots, count, dtype=torch.complex128)
    for index in range(n_slots):
        out[:, index, :] = out[:, index, :] + _mm(
            slots[:, index, :].conj(),
            phi_slots[:, index, :].transpose(-1, -2),
            torch.complex128,
        )
    return out.to(target)


def basis_readout(
    bound: torch.Tensor,
    basis: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """out[b, k] = sum_d bound[b, d] * conj(basis[k, d])."""
    return _mm(bound, basis.conj().transpose(-1, -2), dtype)
