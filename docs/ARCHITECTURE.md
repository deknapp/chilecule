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
│  tools/   chem  alerts  lookup  chembl  sar             │
│           structure  pockets  docking                   │
│  bench/   metrics  decoys                               │
└─────────────────────────────────────────────────────────┘
```

Everything runs on a laptop CPU. There is no cluster tier and no GPU tier.

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
All eleven tools publish an output schema, so a client can destructure the
response instead of re-parsing text.

**Arguments are validated before any work starts.** This is the only place in
the codebase using pydantic, and the narrowness is deliberate. The scientific
layer's inputs come from Python a person wrote and a type checker has already
seen; runtime validation there would buy nothing over the 22 dataclasses
already in use. The agent boundary is different: arguments arrive from a
language model, which is exactly the condition that justifies validating them.

Concretely, `exhaustiveness` is `Field(8, ge=1, le=64)` with a description of
what the parameter costs, `pdb_id` carries a pattern, and `smiles` runs an
actual RDKit parse — because `"aspirin"` is a syntactically valid string and
only a parse attempt catches it. A model passing `exhaustiveness=10000` gets a
correctable error in milliseconds instead of a docking run that never returns,
and there is a test asserting that path stays under two seconds so the check
cannot silently move behind the PDB download.

The constraints also travel to the model in the published JSON Schema. Tool
descriptions and argument schemas are prompt surface; a bare
`{"type": "integer"}` makes the model guess, and the guess costs a docking run.

What this does not do, and it is worth being clear about: it validates shape,
range, and parseability. Not one of the real bugs found while building this
project — the stripped catalytic metal, the matched-pair direction artifact,
the tie handling that gave a constant scorer a perfect enrichment factor —
involved malformed data. Types are the cheap layer. Controls and tests are the
load-bearing one.

---

## Execution

`chilecule/parallel.py` contains one function, `pmap`. That is the whole
execution layer.

An earlier version of this project had a `Runner` protocol with a local
implementation and a scaffolded AWS Batch backend. It has been removed. Every
workflow here is designed to finish on a laptop CPU, so there was exactly one
implementation, and an abstraction over one implementation is a claim about
generality that the code does not cash. Deleting it removed about 200 lines
and a `boto3` dependency without changing what the project can do.

The only judgement left is thread pool versus process pool: RDKit releases the
GIL for much of its C++ work, so RDKit-heavy and I/O-shaped work parallelizes
on threads, while Python-level per-molecule loops need processes.

If GPU-scale work is ever wanted here — structure prediction, free energy
perturbation — the right move is to add it when there is a workflow that needs
it, not to keep an empty backend around in anticipation.


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
