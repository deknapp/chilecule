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

`chilecule doctor` tells you what is available. There is nothing here that
needs a GPU or a cluster — see [Status](#status).

---

## The workflows

Everything runs on a laptop CPU in seconds to minutes. Nothing needs a GPU.

Most tools in this space are *target-in*: name a protein and they go looking.
That is the rarer case. A medicinal chemist on a live program already knows the
target — they arrive holding a molecule, a series, or a list somebody just sent
them. So these come first.

### `chilecule profile compounds.smi`

The daily workhorse. For one compound or five hundred: standardization,
properties, structural alerts, synthetic accessibility, a multi-parameter
scorecard, plus the two questions that are most annoying to answer by hand —
**has anyone made this** (exact InChIKey against ChEMBL and PubChem, with near
neighbours) and **what else does it hit**.

The scorecard reports the *limiting property*, not just a score. `0.42` tells a
chemist nothing; "limited by cLogP at 5.8, target below 4" is a design
instruction.

For promiscuity it reports the **selectivity window** — log units between the
best target and the median of the rest — because the raw target count is
misleading. Erlotinib has 113 reported protein targets and is not promiscuous
in any troubling sense; it is a kinase inhibitor that has been through kinome
panels, and its 3.56-log window says so. A *flat* profile across many unrelated
proteins is what actually indicates an aggregator.

### `chilecule analogs '<SMILES>' --target EGFR`

What should I make next? Proposes analogs built only from transformations
medicinal chemists have already made against that target, mined from ChEMBL
matched molecular pairs, each carrying its evidence:

```
CO[*:1] >> F[*:1]   attached to aromatic C
n_pairs 11 · median +0.43 log · SD 0.31 · well-supported
```

Deliberately not generative. A model will happily propose a molecule nobody has
made for a reason; a transformation observed eleven times, with its spread
shown, is a proposal a chemist can argue with — and arguing with it is the
point. When the parent falls outside the chemical space the evidence came from,
the expected effect reads **no decision** rather than a number.

### `chilecule series data.csv`

For a congeneric series you already have data on: R-group decomposition into a
substituent × position table, Free-Wilson additivity with a **cross-validated**
R² (a one-hot model memorizes a small series completely, so the training fit
always looks excellent), activity cliffs, and the question that gets asked too
late — **is potency being bought with lipophilicity?** If potency correlates
with cLogP across your series, that works right up until solubility, promiscuity
and hERG arrive together in preclinical.

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

Runnable versions, with real output, are in [`examples/`](examples/).

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
  lookup        novelty (ChEMBL/PubChem) and promiscuity across reported targets
  chembl        retrieval + curation with a full audit trail
  sar           scaffolds, matched molecular pairs, activity cliffs
  structure     PDB retrieval, structure ranking, receptor prep, docking boxes
  pockets       fpocket binding-site detection
  docking       smina / vina / gnina, plus the redocking control

bench/          metrics (EF, BEDROC, ROC-AUC) and decoy bias detection
workflows/      the pipelines above, each returning one Report
mcp/            the tool layer exposed over Model Context Protocol
parallel.py     pmap(). The entire execution layer.
```

**The tool layer contains no LLM calls.** Every scientific function is
deterministic, unit-testable, and usable without an API key. The agent layer
sits on top and decides *which* tools to call and *how to interpret* the
results — which pocket to target, when a filter is too aggressive, when a
docking result is not credible. That boundary is deliberate: an agent that can
hallucinate a logP is not useful, and an agent that cannot exercise judgment
about a screening cascade is just a shell script with a chat interface.

### What testing the agent actually found

The tool layer has 89 unit tests. Running the *agent* over real tasks found two
bugs that none of them caught, because every unit test checks a function against
an input I chose.

**It refused to claim a result it didn't have.** Asked to profile aspirin, the
agent reported the properties from tools and then stopped:

> *"No tool in this toolkit resolves a structure to a compound record — the
> ChEMBL tools available take a target query, not a structure or InChIKey. So
> this identification is mine, not a tool's. Treat that as unverified."*

It was right. I had written `check_novelty` and tested it directly in Python,
and never exposed it over MCP. Worse, the underlying cause was that I had
appended the new `@mcp.tool` definitions *after* the `__main__` guard — so an
in-process import registered all seventeen tools and running the server as a
module silently served eleven.

**It caught chemistry I would have shipped.** Asked to design analogs of a
gefitinib-like hit, it inspected the output and flagged:

> *"Three of its 22 proposals contain Ar–O–Br or Ar–O–Cl bonds, produced by
> applying the fragment rule `C[*:1] >> Br[*:1]` to the methyl of a methoxy
> group. A hypobromite ester is not a compound you can make. The Br version
> claims 28 pairs and 'moderate' reliability — the highest pair count in the
> list attached to a physically meaningless product."*

Correct, and the fix was structural: a matched-pair transformation is only valid
in the attachment environment it was mined from. Methyl-to-bromo is ordinary
aromatic substitution on a ring carbon and absurd on an ether oxygen.
Transformations now carry their attachment context as part of their identity,
with a plausibility backstop behind it. Both bugs have regression tests.

**And a third, after the fix.** Running the design task again, the agent
reported that `design_analogs` "errors on every parameter combination I tried —
worth a bug report." It did: adding the attachment context to the
transformation dict without adding it to the corresponding pydantic model made
`extra="forbid"` reject every response. The guard worked exactly as intended and
nothing tested the contract between the two layers, so `tests/test_schema_contracts.py`
now asserts that every model accepts exactly what its dataclass produces.

None of these is a story about the model being clever. They are a story about
the tool layer being wrong in ways that only showed up when something tried to
*use* it for a real task.

There is now a small behavioural eval in [`evals/agent_eval.py`](evals/agent_eval.py) —
six cases checking things a model gets wrong by default, like answering from
memory instead of calling the lookup, or converting a docking score into a Kd.
It makes real API calls, so it is not part of `pytest`.

---

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

**Deliberately absent:** any cloud or GPU tier. An earlier version carried a
scaffolded AWS Batch runner behind a `Runner` protocol; it was deleted. Every
workflow here finishes on a laptop CPU, so the abstraction had exactly one
implementation and was making a claim about generality the code did not cash.

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
