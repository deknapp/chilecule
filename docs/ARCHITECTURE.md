# Architecture

## The one structural decision

**The tool layer contains no LLM calls.**

Everything in `chilecule/tools/` and `chilecule/bench/` is a deterministic
function over molecules, structures, and data frames. Same input, same output,
no API key, no network unless the function's job is to fetch something. Every
one of them is unit-testable, and 73 of the 84 tests run with no network and no
external binaries.

The agent layer sits on top and does not compute anything. It decides *which*
tools to call and *how to read* the results: which of 30 fpocket cavities is
the real site, whether a filter that removed 90% of a library was too
aggressive, whether a docking score means anything given how the control went.

The boundary matters in both directions. An agent that can hallucinate a logP
is worse than useless, because the number looks right. And a pipeline that
cannot exercise judgment about a screening cascade is a shell script with a
chat interface. Putting the science below the boundary and the judgment above
it is what makes each half do the thing it is actually good at.

```
┌─────────────────────────────────────────────────────────┐
│  Agent layer          judgment, interpretation, framing │
│  workflows/  agents/  MCP clients                       │
└────────────────────────────┬────────────────────────────┘
                             │  MCP  (mcp/server.py)
┌────────────────────────────┴────────────────────────────┐
│  Tool layer                deterministic, no LLM        │
│  tools/   chem  alerts  chembl  sar  structure          │
│           pockets  docking                              │
│  bench/   metrics  decoys                               │
└────────────────────────────┬────────────────────────────┘
                             │  Runner
┌────────────────────────────┴────────────────────────────┐
│  Execution      LocalRunner (complete)                  │
│                 AWSBatchRunner (scaffold)               │
└─────────────────────────────────────────────────────────┘
```

---

## Why MCP rather than a bespoke tool registry

Exposing the tools over the Model Context Protocol means they work in Claude
Code, Claude Desktop, or any other MCP client — not only inside this project's
own agent loop. The tools outlive whatever orchestration framework is
fashionable, and a user who wants to do one thing interactively is not obliged
to adopt the whole workflow layer.

Two design rules for the server, both of which cost tokens and both of which
pay for themselves:

**The caveats live in the tool descriptions.** `dock_molecule` states in its own
description that a docking score is not a predicted affinity. A model that has
only seen the signature will confidently report −9.2 as "strong predicted
binding". Putting the limitation where the model reads it is the difference
between a tool that informs and one that misleads.

**Results are structured, with units and interpretation.** A tool returning
`-9.2` invites the model to invent a meaning. One returning
`{"score": -9.2, "units": "kcal/mol", "interpretation": "ranking signal only",
"control_verdict": "protocol reproduces the crystallographic pose"}` does not.

---

## The Runner abstraction

Every expensive step declares a `ResourceProfile` — cores, memory, expected
seconds per item, whether it needs a GPU — and hands work to a `Runner`. The
workflow never knows where execution happened.

```python
runner.map(dock_one, ligands, DOCKING_POSE)
```

The point is that scaling out must not require rewriting the science. Docking
500 ligands on a laptop is the same code that would dock 5 million on a cluster;
only the runner changes.

`LocalRunner` is complete. It picks a process pool for CPU-bound profiles and a
thread pool otherwise, and runs small batches serially because pool startup
costs more than the work below about 32 items.

---

## The cloud tier is a scaffold, on purpose

`AWSBatchRunner` defines the interface and raises `NotImplementedError` from
every execution method. `available()` returns `False` unconditionally, and
`chilecule doctor` reports "AWS credentials detected" separately from "cloud
tier works" — conflating those would tell a user their setup is fine when the
backend does not exist.

This is a deliberate stopping point rather than an unfinished corner. Every
shipped workflow is designed to complete on laptop CPU, so nothing needs it;
and a portfolio repository that leaves billable infrastructure running is worse
than one with no cloud tier at all.

### What implementing it would involve

**1. A container image built from this project's own environment.** Not a
separately maintained cloud Dockerfile — two dependency specifications drift
apart within a month, and the resulting "works locally, fails in the cloud" bug
is expensive and boring. One source of truth or none.

**2. Infrastructure as code.** An S3 bucket for inputs and results, an ECR
repository, a Batch compute environment on Spot, a job queue, and one job
definition per resource tier. A single CloudFormation stack created by
`chilecule cloud init` and destroyed by `chilecule cloud destroy`. Teardown is
not optional.

**3. A cost gate before submission.** `estimate_cost()` already produces the
projection; the runner shows it and requires confirmation above a threshold.
Spot instances by default, with a hard budget cap on the compute environment.
Nothing that spends money should do so without saying how much first.

**4. Content-addressed results.** S3 keys derived from a hash of (input, tool
version, parameters), so re-running an unchanged job is a cache lookup rather
than a second charge. Docking is deterministic given a seed, which makes this
straightforwardly correct.

### Credentials

Not handled by this project, deliberately. Botocore's resolution chain —
environment, shared config profiles, SSO, container and instance roles —
already does this correctly and supports short-lived credentials. The Anthropic
SDK's chain does the same for model access, including OAuth profiles from
`ant auth login`. A project that invents its own credential file is a project
that will eventually leak one, and it would drop SSO and instance-role support
in exchange.

`chilecule/config.py` therefore only *detects* what is present, and reports
presence without ever printing a value. Note also that AWS credentials can
serve both roles at once: Claude runs on Bedrock, so a user with AWS
credentials has both the compute and the model behind one key.

---

## Report objects

Every workflow returns a `Report`, which renders to Markdown for a human and to
JSON for an agent from the same object. Two properties are load-bearing:

**Warnings render above findings.** A caveat placed below a conclusion is a
caveat nobody reads. When the negative-set bias check fails, that failure is the
first thing in the document, above the enrichment table it invalidates.

**Provenance is mandatory.** ChEMBL release, PDB entry, docking program,
platform, package version. A result that cannot say what produced it is not
reproducible, and an irreproducible result in this field is worth nothing.

---

## Adding a workflow

1. Put the science in `tools/` as pure functions with tests.
2. Compose them in `workflows/`, returning a `Report`.
3. Expose anything an agent should call directly in `mcp/server.py`, with the
   caveats written into the description.
4. Add a CLI command in `cli.py`.
5. If it can be validated retrospectively, add it to `bench/` — and if it
   cannot, say so in the report.
