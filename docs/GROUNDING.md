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
| `nlm doctor` (after `--force`) | 54 cookies, account **unknown** |
| network | `google.com` 200, `notebooklm.google.com` 301, sub-second |
| profile `base_host` | `notebook.google.com` |
| CLI version | installed **0.10.1**, latest **0.15.4** |

Diagnosis: `DERIVED`. This is not a credential gap and not a network
fault. The stored session is stale at the NotebookLM API layer, and the
installed client predates the current backend (4 minor versions behind).

Two remedies, both requiring a human decision:

1. `nlm login --manual --file <cookies.txt>` with an exported cookie jar.
2. Upgrade `notebooklm-mcp-cli` to 0.15.4 in a **new session**, then
   re-auth. Mid-session MCP upgrade is out of policy.

Consequence: no notebook synthesis is claimed. The attached specs and
the retired-repo register are the grounding layer for this build.
