# Repository status and lineage

Recorded 2026-10-12. Measured, not narrated.

## Authoritative kernel

**HENRI v6** — `C:/Users/chan/Desktop/HENRI v6` — is the authoritative
kernel. All new work targets this checkout.

## Remote

| Item | Value |
|---|---|
| `origin` | `https://github.com/cjc214foodun9/HENRI-v6.git` |
| Pushed branch | `review/phase3-active-inference` |
| Verified SHA | `55d76fcb5daf975c638cbf1dd02ce48281a70eb2` |
| Verified at | 2026-10-12 |
| Readback | `git ls-remote` SHA == local HEAD, AND a fresh `git fetch` of that branch reproduces the same SHA with 67 files |

## Remote `main` - PROMOTED 2026-10-12 under explicit owner approval

Owner instruction: "please promote to main. i provide explicit approval."

**Recovery artifacts, created and verified BEFORE the destructive step:**

| Artifact | Value |
|---|---|
| Backup bundle | `%LOCALAPPDATA%/Temp/hv6_backups/full-20261012-201241.bundle`, 218440 bytes, `git bundle verify` ok, 7 heads, carries the old tip |
| Earlier bundle | `%LOCALAPPDATA%/Temp/hv6_backups/henri-v6-backup-20261012-200838.bundle`, 214886 bytes |
| Backup ref | `backup-remote-main-6955077` -> `695507707d883e1d11abe8df9360417993552f88` |
| Old tip object | present locally, so recovery does not depend on the network |

**The old history was ARCHIVED ON THE REMOTE, not destroyed.** Before
`main` moved, the previous tip was pushed to a new remote branch:

```
695507707d883e1d11abe8df9360417993552f88 -> refs/heads/archive/manual-upload-2026
```

That branch is still on GitHub. The earlier upload remains fully
retrievable by anyone, so this promotion is reversible without the local
backups.

**Push method: `--force-with-lease` pinned to the old tip.** The lease
would have aborted the push had any other writer moved `main` between the
read and the write. It succeeded.

**Post-push readback** (`git ls-remote`, not the push message):

| Ref | SHA |
|---|---|
| `refs/heads/main` | `1b80961fde76ac8d7c36941c4770629a8bdbfc80` = local HEAD |
| `refs/heads/archive/manual-upload-2026` | `695507707d883e1d11abe8df9360417993552f88` |
| `refs/heads/review/phase3-active-inference` | `1b80961fde76ac8d7c36941c4770629a8bdbfc80` |

Content comparison against the superseded remote `main`:

| File | Status |
|---|---|
| `binding.py`, `config.py`, `gram.py`, `koopman.py`, `langevin.py`, `pipeline.py`, `sagnac.py` | **byte-identical** |
| `hopfield.py`, `kuramoto.py`, `memory.py`, `readout.py`, `state.py` | **differ** - exactly the five files repaired in this session |

The five differing files are the repaired ones, so the promotion carries
the session's fixes rather than discarding them.

## Retired repositories

The following are RETIRED research inputs. They are read-only reference:
no new work targets them, and nothing is deleted from them.

| Path / URL | Status |
|---|---|
| `C:/Users/chan/Desktop/HENRI 7B SWARM` | RETIRED — remote `https://github.com/cjc214foodun9/HENRI.git`, 150 branches |
| `HENRI V2` | RETIRED — legacy primitive source |
| `https://github.com/cjc214foodun9/HENRI.git` | RETIRED — do not push new work here |

**Archival is by declaration, not deletion.** This checkout holds no
branches of those repositories. A branch operation inside `HENRI v6`
cannot archive them, because they live in a different checkout. Nothing
was deleted or rewritten. The retired repositories remain available in
full, and their verified primitives were re-derived rather than copied.

Rationale for not vendoring the retired code: the audit found its
readout saturating at chance and its `epsilon = 0.35` hardcoded for the
wrong wave family. Primitives were re-derived from the specification and
verified against their own gates. See `docs/EVIDENCE_REGISTER.md`.


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
