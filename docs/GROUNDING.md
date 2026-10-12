# Grounding register

## Sources actually used

| Source | Kind | Use |
|---|---|---|
| `Downloads/Project HENRI.md` | attached spec | tri-model topology, mem-65M, swarm, decoder |
| `Downloads/henri_wave_mechanical_model_scaffolding_specification.md` | attached spec | levels 0-5, gates G-U1..G-U7 |
| `Downloads/project_henri_forensic_discrepancy_probe_...specification.md` | attached spec | five ML failures, directives 1-5 |
| retired repo audit register | OBSERVED receipt | constraint list C1-C16 |
| GitHub `cjc214foodun9/HENRI` README | OBSERVED | evidence policy, status |

## NotebookLM — `BLOCKED`

Target bank: `ca4bb787-de9d-4ee0-89c9-bf71259cc86d`.

Measured 2026-10-12:

| probe | result |
|---|---|
| `nlm login --check` | "Authentication valid" - **false pass** (probes homepage) |
| `nlm notebook list` | `ClientAuthenticationError` |
| MCP `notebook_get` | `Authentication expired` |
| `nlm doctor` (before) | 63 cookies, CSRF yes, account `chandler.coleman34@gmail.com` |
| `nlm doctor` (after `--force`) | 54 cookies, CSRF yes, account **unknown** |
| profile `metadata.json` (after `--force`) | `email` field **empty**; `csrf_token` present; `last_validated` fresh |
| `nlm login --force` | reported "✓ Successfully authenticated!", 41 cookies |
| `nlm notebook list` (after `--force`) | still `ClientAuthenticationError`; `rc=1` |
| profiles on disk | **1 real** (`default`); 2 empty dirs `__nonexistent_probe__`, `does_not_exist` |
| network | `google.com` 200, `notebooklm.google.com` 301, sub-second |
| CLI version | installed **0.10.1**, latest **0.15.4** |

### Hypotheses tested and FALSIFIED

`OBSERVED`. Two candidate root causes were measured and rejected.

1. **Stale cookies - FALSIFIED.** After `--force` the profile holds 54
   freshly extracted cookies, a valid CSRF token, and a fresh
   `last_validated`. The data call still raises
   `ClientAuthenticationError`. Fresh credentials plus a failing data
   call is not a credential-staleness signature.
2. **A second usable profile - FALSIFIED.** `doctor` reports
   "Profiles found: 3". Only `default` is real. The other two entries
   are empty directories (`__nonexistent_probe__`, `does_not_exist`)
   that hold no `metadata.json` and no `cookies.json`. No alternative
   account binding exists to try.

### Consequence of `login --force` - OBSERVED regression

The command reported success and made the state **strictly worse**. It
changed the recorded account from `chandler.coleman34@gmail.com` to
empty, and `doctor` now reports `Account: unknown`. It did not fix data
access. Label `login --force` a **false-pass remedy**: it validates
cookie extraction, not the API call.

Do not run it again. Three attempts produced identical failures and the
account binding is already lost.

### Remaining hypothesis - not yet tested

`HYPOTHESIS`. Client version drift. The installed client is 0.10.1.
The recorded `build_label` is
`boq_labs-tailwind-frontend_20261011.03_p...`, a Google frontend build
from 2026-10-11. A moved backend contract would produce exactly this
signature: valid cookies, valid CSRF, rejected RPC. This is untried.

### Remedies, both requiring a human decision

1. Upgrade `notebooklm-mcp-cli` to 0.15.4 in an **isolated venv**,
   against a **backed-up copy** of the profile directory, then re-auth.
   The `default` profile is already degraded; back it up first.
2. `nlm login --manual --file <cookies.txt>` with an exported cookie jar.

Consequence: no notebook synthesis is claimed. The attached specs and
the retired-repo register are the grounding layer for this build.
