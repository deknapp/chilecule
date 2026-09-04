# chilecule

**Agentic drug discovery workflows built entirely on open-source tooling.**

RDKit, AutoDock Vina / smina, fpocket, ChEMBL, the PDB — wrapped as MCP tool
servers, driven by LLM agents, and checked by a validation harness that will
tell you when the results are meaningless.

[![CI](https://github.com/deknapp/chilecule/actions/workflows/ci.yml/badge.svg)](https://github.com/deknapp/chilecule/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Dependencies](https://img.shields.io/badge/deps-license--audited-brightgreen.svg)](docs/LICENSING.md)

---

## Why another one of these

There are already a lot of repositories that connect an LLM to RDKit. Most of
them compute a logP, print it, and stop. What is almost always missing is any
evidence that the pipeline *works* — and in this field, plausible-looking output
is very easy to generate and very hard to trust.

So the organizing principle here is that **every claim the tooling makes has to
be checkable, and the tooling has to be willing to say when it is wrong.** Three
examples, all real output from this repository:

**The docking protocol validates itself, and caught a bug in my own code.**
Before docking anything, `chilecule` re-docks the co-crystallized ligand into
its own structure and measures RMSD against the crystallographic pose. On EGFR
structure 5CNN this failed at 2.87 Å — because `prepare_receptor` was
discarding Mg²⁺ along with the crystallization additives. Magnesium coordinates
the phosphates in every kinase nucleotide site; removing it deletes the
electrostatics the ligand binds to. Retaining catalytic metals takes the same
control to **1.54 Å — pass**. Without the control, that bug ships silently and
every docking score downstream is quietly wrong.

**The benchmark checks itself for bias before reporting a number.** Retrospective
validation needs negatives. `chilecule` trains a classifier on physicochemical
descriptors alone — no fingerprints, no protein — and tries to separate actives
from negatives. On EGFR's *experimentally confirmed* ChEMBL inactives it reaches
**AUC 0.955**, so the harness reports `FAIL` and states that the enrichment
figures do not demonstrate the method works. Real experimental negatives are not
automatically fair negatives.

**Enrichment is always printed next to its ceiling.** An EF of 14 means nothing
on its own. On EGFR, ECFP4 similarity search scores **EF₁% = 14.2 ± 7.4 against
a theoretical maximum of 48.5**, with random at 0.7 ± 2.1 — measured over 20
replicates subsampled to a realistic 2% active fraction, because ChEMBL yields
~50% actives and on such a set even a perfect method cannot post a meaningful
enrichment factor.

Every number above, plus the commands to reproduce it, is in
**[docs/VALIDATION.md](docs/VALIDATION.md)** — including a section on what is
*not* validated.

---

## Install

```bash
git clone https://github.com/deknapp/chilecule
cd chilecule
./install.sh              # RDKit + Python layer, ~2 minutes
./install.sh --tier dock  # adds smina, fpocket via conda-forge
chilecule doctor          # reports what is available and which tiers can run
```

`install.sh` uses micromamba for the binaries that are not pip-installable and
`uv` for the Python layer. It never touches your global environment and prints
what it is about to do before doing it.

### Credentials

There is no config file and nothing to paste. The SDKs already resolve
credentials correctly, so `chilecule` reads their chains and reports what it
found:

| You have | You get |
|---|---|
| Nothing | Every workflow below. All of them run on laptop CPU. |
| `ANTHROPIC_API_KEY`, or `ant auth login` | Agent-driven workflows |
| AWS credentials + `CLAUDE_CODE_USE_BEDROCK=1` | Agent-driven workflows via Bedrock |

`chilecule doctor` tells you which tier you are in. Anything requiring a GPU is
scaffolded but not implemented — see [Status](#status).

---

## The workflows

All four run on a laptop CPU in seconds to minutes. None require a GPU.

### `chilecule dossier EGFR`

What is publicly known about a target, reconciled across UniProt, the PDB, and
ChEMBL — the assay landscape, the structures worth docking into, the known
chemotypes, and the property envelope that has already produced activity.

Structures are ranked for **docking suitability, not sequence coverage**: a
1.9 Å crystal structure of the kinase domain beats a 3.1 Å cryo-EM model of the
full-length receptor, however much of the sequence the latter spans.

### `chilecule sar EGFR`

Matched molecular pair analysis, scaffold decomposition, and activity cliffs.
This answers the question a project team actually asks — *what did changing that
group do* — as a table of transformations with median potency deltas and the
spread across contexts.

Activity cliffs are reported because they carry the most information per
compound in any dataset, and because they bound what any QSAR model can achieve:
a model cannot be simultaneously smooth and correct across a cliff.

### `chilecule triage library.smi --pdb 5CNN`

A filter cascade, cheapest stage first, ending in docking. Every excluded
compound keeps the reason it was excluded.

Docking is **gated on the redocking control**. If the protocol cannot reproduce
a known crystallographic pose for that site, the report says the scores are not
evidence rather than ranking on them anyway.

### `chilecule validate EGFR`

The one that makes the others answerable. Checks the negative set for bias,
then measures ROC-AUC, BEDROC, and enrichment factors against their ceilings,
with a random baseline and a drug-likeness-only control.

Runnable versions of all four, with real output, are in
[`examples/`](examples/).

---

## Data curation

ChEMBL is the backbone of open drug discovery and also where naive pipelines
quietly produce garbage. `curate_activities()` applies the checks that separate
a defensible SAR table from pooled noise, and reports every removal:

```
ChEMBL curation for CHEMBL203 (ChEMBL_37)
  fetched 4000 activity records
  -   168  curator flag: data_validity_comment marks record as suspect
  -   276  endpoint type: standard_type not in ['IC50', 'Ki']
  -   743  censored: standard_relation is an inequality, not a point estimate
  -    57  unconvertible units: standard_units not a molar concentration
  -     8  implausible ligand efficiency: LE > 0.83 kcal/mol/heavy atom
  -   408  inconsistent replicates: measurements disagree by >1.0 log units
  = 2340 records over 1884 unique compounds
```

Each of those lines is a bug someone has shipped:

- **Censored values.** `>10000 nM` records carry a `standard_value`, and a
  regression model will happily learn that every inactive compound has exactly
  10 µM potency. They are recoverable separately as a labelled negative set.
- **Ligand efficiency ceiling.** A 10-heavy-atom fragment reported at 0.1 nM
  implies LE of 1.36 kcal/mol per atom, which no organic fragment achieves. This
  catches transcription errors that pass both unit checks and range checks.
- **Replicate disagreement.** A compound with reported IC50s of 30 nM and 8 µM
  has not been measured; it has been measured twice in incompatible assays.
- **Species mixing.** Human and rodent orthologs differ enough at the binding
  site that pooling them is a modelling error, not a data-volume win.

Every curated frame carries its ChEMBL release for CC BY-SA attribution.

---

## Architecture

```
tools/          Pure functions over molecules and structures. No LLM anywhere.
  chem          standardization, descriptors, ligand & lipophilic efficiency
  alerts        PAINS / Brenk / NIH / ZINC, annotated with severity — never deleted
  chembl        retrieval + curation with a full audit trail
  sar           scaffolds, matched molecular pairs, activity cliffs
  structure     PDB retrieval, structure ranking, receptor prep, docking boxes
  pockets       fpocket binding-site detection
  docking       smina / vina / gnina, plus the redocking control

bench/          metrics (EF, BEDROC, ROC-AUC) and decoy bias detection
workflows/      the four pipelines above, each returning one Report
runners/        LocalRunner (complete) and AWSBatchRunner (scaffold)
mcp/            the tool layer exposed over Model Context Protocol
```

**The tool layer contains no LLM calls.** Every scientific function is
deterministic, unit-testable, and usable without an API key. The agent layer
sits on top and decides *which* tools to call and *how to interpret* the
results — which pocket to target, when a filter is too aggressive, when a
docking result is not credible. That boundary is deliberate: an agent that can
hallucinate a logP is not useful, and an agent that cannot exercise judgment
about a screening cascade is just a shell script with a chat interface.

Exposing the tools over MCP means they work in Claude Code, Claude Desktop, or
any MCP client — not only inside this project's own agent loop:

```bash
chilecule serve-mcp    # then point any MCP client at it
```

All eleven tools are typed on both sides. Inputs are validated before any work
starts, and the constraints travel to the model in the published schema:

```python
exhaustiveness: Annotated[int, Field(8, ge=1, le=64, description=
    "Docking search effort. Runtime scales roughly linearly. 8 is the screening "
    "default; 16-32 when a single pose matters. Above 32 the returns are "
    "negligible and the cost is not.")]
```

A model that passes `exhaustiveness=10000` gets a correctable error in
milliseconds rather than a docking run that never returns. `smiles` runs an
actual RDKit parse, because `"aspirin"` is a syntactically valid string and only
a parse attempt catches it — the error says so, so the model can fix it on the
next turn.

This is the only place in the codebase that uses pydantic, and the narrowness is
the point: it is the one boundary where arguments come from a language model
rather than from a programmer. The scientific layer uses plain dataclasses.

---

## Licensing

The core is Apache-2.0 and depends on **no GPL or LGPL library**. Every
dependency was audited by hand; the full analysis, including the reasoning about
process isolation and data licensing, is in
**[docs/LICENSING.md](docs/LICENSING.md)**.

| Dependency | License | How it is used |
|---|---|---|
| RDKit | BSD-3-Clause | imported |
| NumPy, pandas, scikit-learn | BSD-3-Clause | imported |
| requests | Apache-2.0 | imported |
| smina / AutoDock Vina / gnina | Apache-2.0 | **subprocess only** |
| fpocket | MIT | **subprocess only** |
| Open Babel *(optional)* | GPL-2.0 | **subprocess only**, never required |

Docking binaries are invoked as external processes over files — never imported,
never linked — so their licenses, and those of anything they link, do not reach
this codebase. conda-forge's smina build links Open Babel (GPL-2.0), which is
exactly why that boundary matters.

**Data:** ChEMBL is CC BY-SA 3.0, so it is queried at runtime and no derived
dataset is redistributed here. The PDB is CC0 and PubChem is public domain.
DrugBank is CC BY-NC and is deliberately not wired in.

---

## Status

This is a working v0.1, not a finished product. What is honest to say about it:

**Implemented and verified against real data:** all four workflows, the full
tool layer, the validation harness, the MCP server, the local runner. 92 tests,
of which 81 run with no network and no external binaries; the rest exercise
live ChEMBL, PDBe, RCSB, smina and fpocket.

**Scaffolded, not implemented:** the AWS Batch runner. The `Runner` interface
is fixed and documented, and `docs/ARCHITECTURE.md` describes what building the
cloud tier involves. It is a scaffold on purpose — every shipped workflow is
designed to finish on laptop CPU, and a portfolio repository that leaves
billable infrastructure running is worse than one with no cloud tier at all.

**Not attempted:** free energy perturbation, retrosynthesis, generative design,
and structure prediction. All are interesting; none run on a laptop.

---

## Contributing

Issues and pull requests welcome. The one firm rule: **no new dependency without
a license check**, and no GPL/LGPL library imported into the core. If a tool is
only available under a copyleft license, wrap it as a subprocess and document
the boundary in `docs/LICENSING.md`.

## Citation

This project is a wrapper around other people's work. If you publish using it,
cite the underlying tools — RDKit, AutoDock Vina, fpocket, ChEMBL — not this
repository.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
