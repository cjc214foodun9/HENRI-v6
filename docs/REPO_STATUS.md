# Repository status and lineage

Recorded 2026-10-12. Measured, not narrated.

## Authoritative kernel

**HENRI v6** — `C:/Users/chan/Desktop/HENRI v6` — is the authoritative
kernel. All new work targets this repository.

| Item | Value |
|---|---|
| HEAD at record time | `6abffb0` |
| Remote | none configured (local-only) |
| License | "Diverse Agency" License, sealed 2026-10-12 |
| Suite tier 1 | CPU, `D=512` |
| Suite tier 2 | production `D=65,536`, marker `slow` |

## Retired lineage

The following are **RETIRED**. They remain on disk, read-only by intent.
They are historical record and evidence, not build targets.

| Surface | Location | Status |
|---|---|---|
| `HENRI 7B SWARM` | `C:/Users/chan/Desktop/HENRI 7B SWARM` | RETIRED |
| `HENRI V2` | inside the retired checkout, and its worktrees | RETIRED |
| `github.com/cjc214foodun9/HENRI` | remote | RETIRED |

Retired HEAD at record time: `fa31da4`, branch `carrier/e6-physical-verifier`.

### What "archive" means here, and what it does not mean

The retired branches (`carrier/*`, `accuracy/*`, `backup/*`) live in the
**retired checkout**, not in v6. No branch operation inside v6 can archive
them, because they are not v6's branches.

This record archives them by **declaration**: their status is retired and
this document names the retired HEAD. It does **not** delete them, rewrite
their history, or force-push the retired remote.

Reason: the retired checkout holds preserved user work and dirty worktrees.
Deleting it would destroy evidence this project still cites. Deletion needs
a separate explicit owner approval, and this task grants none.

## Precedence rule

When v6 and the retired checkout disagree, **v6 governs**. The retired
checkout is evidence about the past, not a specification for the present.

Two exceptions. First, the retired audit register still supplies the
16 constraint identifiers that v6's tests pin. Second, the retired
`LICENSE.md` supplied the sealed license text. Both are cited with hashes.

## Non-claims

- v6 is not a production release. It is a verified CPU kernel.
- Pinning v6 as authoritative is a local declaration. No remote enforces it.
- No benchmark score, capability, or hardware claim attaches to this record.

Evidence label: `OBSERVED` for paths, HEADs, and the absence of a remote.
