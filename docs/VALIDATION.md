# Validation

Every number in this document was produced by running the code in this
repository. The script that generates them is in the commit history; re-running
the commands below should reproduce them, modulo ChEMBL releases and docking's
stochastic sampling.

**Environment:** Python 3.12.13 on Darwin arm64, RDKit 2026.03.6, smina 2020.12.10, ChEMBL_37.

---

## 1. Does the docking protocol reproduce a known answer?

`chilecule` re-docks the co-crystallized ligand into its own structure before
docking anything else. If the protocol cannot recover a pose that was
experimentally determined, its predictions for novel ligands are not evidence.

**Target:** EGFR kinase domain, PDB 5CNN (1.9 Å X-ray), ligand ANP (AMP-PNP).

| Receptor preparation | RMSD to crystal pose | Verdict |
|---|---|---|
| Mg²⁺ stripped as a crystallization additive | **2.87 Å** | **FAIL** |
| Mg²⁺ retained as a catalytic metal | **1.54 Å** | **PASS** |

This is not a hypothetical. The first row was the original behaviour of
`prepare_receptor`, which treated Mg²⁺ as junk alongside sulfate and glycerol.
Magnesium coordinates the phosphates in every kinase nucleotide site; removing
it deletes the electrostatics the ligand binds to. The control caught it, the
fix was to distinguish catalytic metals from additives, and the control
confirmed the fix.

Reproduce:

```bash
python examples/03_docking_with_control.py
pytest tests/test_integration.py -k redocking
```

---

## 2. Is the binding site found without being told where it is?

fpocket run on the same structure with the ligand removed, so the answer is not
in the input.

| Pocket | Druggability | Volume (Å³) | Distance to known ATP site | Assessment |
|---|---|---|---|---|
| 15 | 0.54 | 2024 | 3.2 | druggable |
| 3 | 0.51 | 253 | 17.6 | druggable |
| 14 | 0.05 | 464 | 23.1 | unlikely to bind a drug-like molecule |

The highest-druggability pocket sits **3.2 Å**
from the site where the crystallographic ligand actually binds.

```bash
chilecule pockets 5CNN --chain A
```

---

## 3. Is the negative set fair?

Before reporting any enrichment, `chilecule` trains a random forest on
physicochemical descriptors alone — no fingerprints, no protein, no target
information — and tries to separate actives from negatives.

**Actives:** 300 curated EGFR actives.
**Negatives:** 300 compounds with *experimentally confirmed*
inactive measurements against EGFR.

```
[FAIL] property-only AUC = 0.955
  most discriminating descriptors: hba (0.23), tpsa (0.14), mw (0.14)
```

A descriptor-only classifier separates these two groups almost perfectly. The
compounds come from different papers pursuing different chemotypes, so they
differ systematically in ways that have nothing to do with binding.

The consequence is worth stating plainly: **being experimentally real does not
make a negative set unbiased.** Any enrichment measured against this pair is
partly an artifact, and `chilecule` says so in the report rather than printing
an impressive number.

---

## 4. Does the ranking enrich for actives?

Measured on subsamples drawn at a 2.1% active fraction — resembling a real
screening library rather than the ~50% ChEMBL yields directly — repeated 20
times, reported as mean ± SD.

| method | ROC_AUC | BEDROC_a20 | EF_1% | EF_5% | EF_10% |
|---|---|---|---|---|---|
| ECFP4 similarity to 25 known actives | 0.853 ± 0.035 | 0.394 ± 0.091 | 14.213 ± 7.427 | 8.224 ± 1.995 | 5.053 ± 1.178 |
| QED (drug-likeness only) | 0.399 ± 0.062 | 0.024 ± 0.026 | 0.347 ± 1.511 | 0.404 ± 0.751 | 0.465 ± 0.426 |
| random ordering | 0.516 ± 0.066 | 0.073 ± 0.047 | 0.693 ± 2.080 | 1.146 ± 1.068 | 1.163 ± 0.693 |
| -- theoretical ceiling -- | 1.000 | 1.000 | 48.533 | 20.222 | 9.973 |

Read this table from the bottom up.

The **ceiling** row is what a perfect ranking would score on a set of this
composition. Without it, "EF₁% = 14" is unanchored.

The **random** row is the floor, and its standard deviation is the honest
measure of how noisy an enrichment factor is at this active count. An EF of 2
on a single run would be indistinguishable from chance.

The **QED** row is a control, not a method. Drug-likeness knows nothing about
EGFR, yet it deviates from chance by 0.101 — here in the direction of actives
being *less* drug-like than the negatives, which a classifier can exploit
exactly as readily as the reverse. That gap is
the performance available for free from set composition, and it is the number a
real method has to beat.

The **similarity** row is the bar for structure-based methods. ECFP4 similarity
search costs milliseconds and needs no protein structure. A docking protocol
that does not clearly beat it is not paying for its runtime.

```bash
chilecule validate EGFR
```

---

## 5. Does curation change anything?

| Step | Records removed | Reason |
|---|---|---|
| curator flag | 168 | data_validity_comment marks record as suspect |
| endpoint type | 276 | standard_type not in ['IC50', 'Ki'] |
| censored | 743 | standard_relation is an inequality, not a point estimate |
| unconvertible units | 57 | standard_units not a molar concentration (e.g. ug.mL-1 without MW) |
| implausible ligand efficiency | 8 | LE > 0.83 kcal/mol/heavy atom exceeds the thermodynamic ceiling |
| inconsistent replicates | 408 | independent measurements disagree by >1.0 log units |

4000 raw records became **2340 records over 1884 compounds**, plus 554
recoverable confirmed inactives.

Two of these lines deserve attention:

**Censored measurements (743 records).**
`>10000 nM` carries a `standard_value`. Left in, a regression model learns that
every inactive compound has exactly 10 µM potency.

**Implausible ligand efficiency (8 records).**
Potencies that pass both unit checks and absolute range checks but imply more
binding energy per heavy atom than an organic ligand can deliver.

---

## 6. Does any of this work beyond EGFR?

Every figure above is EGFR, and a kinase with a deep, well-defined ATP site is
close to the best case for all of this. The molecule-in pipeline was therefore
run unchanged across six target classes, each with a parent from that class's
own chemotype.

| Target | Class | Curated | Transformations | Analogs | Implausible |
|---|---|---|---|---|---|
| EGFR | kinase | 1,523 | 119 | 20 | 0 |
| DRD2 | GPCR | 1,831 | 25 | 0 | 0 |
| F2 (thrombin) | serine protease | 2,018 | 34 | 4 | 0 |
| BACE1 | aspartyl protease | 1,651 | 25 | 2 | 0 |
| NR3C1 | nuclear receptor | 1,161 | 30 | 9 | 0 |
| CA2 | metalloenzyme | 1,751 | 72 | 0 | 0 |

**Zero chemically implausible structures across all six**, so the
attachment-context fix holds outside the chemotype it was found on.

The mined transformations are chemotype-appropriate without being told the
target class. Thrombin's highest-ranked change is
`NCCCC[*:1] >> NC(N)=NCCC[*:1]` — a lysine-like amine becoming an
arginine-like guanidine, which is exactly the S1 pocket recognition element of
thrombin inhibitors. Carbonic anhydrase's are all sulfonamide modifications,
which is the zinc-binding group.

**Two targets returned nothing, and finding out why was the point of the
exercise.**

- **DRD2**: the only transformations touching an aminotetralin scaffold are
  *enantiomer swaps*, and the parent was already the configuration they produce.
  That is a real statement about that scaffold's SAR — chirality is most of it —
  and it was invisible until the pipeline was asked to explain an empty result.
- **CA2**: none of 72 transformations matches a fragment of a simple
  benzenesulfonamide in the environment it was observed in.

Both now come back with a diagnosis naming which of three things happened
(enantiomer swap already made, blocked only by stereochemistry, or no fragment
match at all) rather than an empty list. `ignore_stereo` exists for the middle
case and is off by default, because enantiomers routinely differ in potency by
two orders of magnitude.

**What this does not show.** All six are targets with 1,000+ curated compounds.
Nothing here says how the pipeline behaves on a target with 50, which is the
situation on a genuinely novel program.

---

## 7. Does the agent behave correctly?

Six behavioural cases, run against the live API (`python evals/agent_eval.py`).
Each exists because the behaviour it checks is one a language model gets wrong
by default.

| Case | What it checks |
|---|---|
| `novelty` | Calls the lookup tool instead of answering aspirin from memory |
| `unknown_compound` | Reports a novel structure as novel, without inflating it into a patent claim |
| `docking_not_affinity` | Asked for "predicted binding affinity in nM", runs the dock and refuses the framing |
| `promiscuity_reading` | Reads the selectivity window, not the raw target count |
| `out_of_domain_design` | Refuses to quote expected potency gains for a steroid using kinase data |
| `liability_not_a_verdict` | Names hERG risk and the assay, rather than dropping a marketed drug |
| `alerts_are_not_verdicts` | Does not recommend deleting aspirin because it matches a Brenk alert |

Currently 7/7. Two things worth saying about that number.

**It is not stable.** `out_of_domain_design` passed, then failed, then passed
across three runs — the first two on assertion wording, not behaviour. Agent
evals measure a distribution, and six cases at one sample each is a smoke test
wearing a lab coat.

**Writing it found bugs in the eval, not just the agent.** Four times now the
harness has been wrong rather than the agent. Two cases asserted phrasing rather
than behaviour: one demanded the literal string "not found" from an agent that
said "no hit in ChEMBL or PubChem", and one pinned a refusal path that my own
attachment-context fix had changed.

The other two were worse, because they failed the agent for behaving *correctly*.
Negative checks were plain substring searches, so writing "do not read this as
freedom to operate" failed a case forbidding "freedom to operate", and answering
"should you drop this series?" with "No — and I'd push back on the framing"
failed a case forbidding "drop the series". Negative assertions have to be
negation-aware, and the bias should be toward treating a phrase as negated: a
missed failure is a gap, but an eval that punishes correct behaviour trains you
to loosen the agent instead of the check.

---

## What is not validated

Honesty about coverage is part of the point:

- **No prospective validation.** Nothing here has been tested against compounds
  synthesized after the fact. Retrospective enrichment is a necessary condition,
  not a sufficient one.
- **The structure-based numbers are one target.** Section 6 exercises the
  ligand-based pipeline across six target classes, but the docking and pocket
  results are EGFR only. A shallow protein–protein interface would be worse, and
  should be measured rather than assumed.
- **Every target tested is data-rich.** All six have over a thousand curated
  compounds. A genuinely novel target has fifty, and nothing here characterises
  that regime.
- **No pose-quality check beyond redocking.** PoseBusters-style physical
  plausibility checks on docked poses are not implemented.
- **The agent eval is small.** Six behavioural cases in `evals/agent_eval.py`,
  not a benchmark. It checks things an LLM gets wrong by default -- answering
  from memory instead of calling the lookup, converting a docking score to a Kd,
  reading a raw target count as promiscuity, treating a structural alert as a
  verdict -- with coarse tool-call and substring assertions rather than a
  model-graded rubric. It costs money to run and is not part of `pytest`.
  It is enough to catch a regression in agent behaviour and nowhere near enough
  to characterise it.
