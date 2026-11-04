# The HENRI v6 command line and the six-holon graph

Status: OBSERVED on CPU, 2026-11-03. This document covers the interface the
DeepSeek Harness plugin calls, the scientific graph it runs, and the immutable
ontology layer it updates.

## 1. Why this module exists

The Harness plugin @henri-holon@ is a thin adapter. It holds no research logic.
Every tool shells out to @python -m henri <command> --json@ and reads one JSON
document. That contract is what makes an agent action reproducible in a
terminal with the same flags.

@RND AGENT/@ is the junctioned research harness at
@C:\Users\chan\Desktop\HENRI v6\RND AGENT@. The DSH profile links the plugin
from @harness/plugins/henri-holon@ inside it. This revision repoints the
plugin's @repoRoot@ to the HENRI v6 checkout, so the plugin drives the code
that is under active development.

## 2. The eight commands

Every command accepts @--json@ and prints:

~~~json
{"ok": true, "exitCode": 0, "summary": "READY", "result": {...}}
~~~

@exitCode: 1@ means a gate failed and @result@ names the cause.

| Command | Purpose | Failure it reports |
|---|---|---|
| @doctor@ | environment and evidence readiness | a missing dependency or a broken chain |
| @ste100@ | ASD-STE100 linter over a file or a string | semicolons, perfect tense, passive voice, banned words |
| @jev@ | evidence integrity gate | a receipt whose SHA-256 does not match |
| @holon@ | run the six-holon scientific graph | a holon that cannot stand on its evidence |
| @gates@ | the four promotion gates | the first gate that fails |
| @ledger@ | run history, leaderboard, chain integrity | a broken ontology chain |
| @zone-check@ | zone write contract | a claim with no receipt, a module with no test |
| @vast@ | GPU executor | a live rental without explicit permission |

~~~powershell
python -m henri doctor --json
python -m henri ste100 --file docs/HOLARCH.md --mode descriptive --json
python -m henri jev --json
python -m henri holon --problem "close gate D4" --profile quick --json
python -m henri gates --profile quick --json
python -m henri ledger leaderboard --json
python -m henri zone-check --path henri/cli.py --zone 2 --json
python -m henri vast --json
~~~

## 3. The six holons

One holon is one stage with one job. The graph stops at the first failure,
because a later stage that inherits a broken fact produces a confident wrong
answer.

| Holon | Job | What it produces |
|---|---|---|
| FRAME | restate the goal as a claim that can fail | the claim text and a testability verdict |
| MEASURE | run receipt integrity, documentation coverage, and the runtime check | counts, exit codes, evidence map |
| FALSIFY | corrupt a copy of a real sidecar and require the checker to reject it | proof that the gate can fail |
| ATTRIBUTE | tie every receipt to its SHA-256 | receipt to digest map |
| RECORD | append immutable ontology entries and a ledger row | chain length and head |
| GATE | decide promote or block | four named gates |

The falsify stage is the reason the graph is not a rubber stamp. A gate that
has never rejected anything is not a gate.

## 4. The immutable ontology layer

@data/holarch/ontology.jsonl@ is append-only and hash-chained. One line is one
entry:

~~~text
{seq, ts, kind, subject, predicate, object, evidence, prev, digest}
~~~

@digest@ is the SHA-256 of the entry body including @prev@. No entry is ever
rewritten. A new fact is a new line. Verify the chain with:

~~~powershell
python -m henri ledger verify --json
~~~

An edited or reordered line fails at its own index, and the verifier names it.
The test suite proves this by rewriting a real row and requiring a failure.

## 5. The orchestrator

@henri.orchestrator.Orchestrator@ is the policy layer. It chooses a plan from
the goal text, runs the graph, writes one ledger row, and returns the verdict.

| Plan | Trigger words | Profile | Reason |
|---|---|---|---|
| promotion | gate, promote, close, D1, D3, D4, release | full | a promotion claim needs the test suite |
| measurement | measure, benchmark, noise, accuracy, receipt | quick | receipt integrity plus documentation coverage |
| release | commit, push, publish, version | quick | a clean evidence chain before anything leaves the host |
| survey | anything else | quick | default survey |

Autonomy is bounded on purpose. The plan chooses a profile and a timeout, never
arbitrary code. The only external actions are registered commands. A failure
stops the graph. Every run is recorded before the answer returns.

## 6. Pointing the plugin at this repository

The patch file sets one key:

~~~yaml
- insert:
    - id: henri-holon
      name: '@henri/dsh-henri-holon'
      config:
        repoRoot: 'C:\Users\chan\Desktop\HENRI v6'
~~~

The adapter runs @python -m henri <command> --json@ with @cwd@ set to
@repoRoot@, so the package resolves from the checkout itself.

Activation is host side. Install or reinstall the bundle, then restart the
application:

~~~text
plugin_manager action=install_bundle
  target=C:\Users\chan\Desktop\HENRI v6\RND AGENT\harness\plugins\henri-holon
~~~

Confirm the row is active, then call a tool:

~~~text
cordis_inspect_query provider=Config method=listConfigs
  input={"name": "@henri/dsh-henri-holon"}
~~~

## 7. What is not claimed

* No GPU run. @vast@ is a dry run that rents nothing.
* The ontology proves integrity, not truth. It records what the evidence says.
* @zone-check@ is a deterministic heuristic, not a proof of correctness.
* The plugin is a host component. This repository supplies the CLI and the
  graph; the host decides when to load it.
