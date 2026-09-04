# Licensing audit

Every dependency in this project was checked by hand against its upstream
`LICENSE` file or its GitHub API license field, not against a summary site.
This document records what was found, what it implies, and where the boundaries
are drawn.

**Summary: `chilecule` is Apache-2.0 and imports no GPL- or LGPL-licensed
library.** Copyleft tools are usable and are used, but only as separate
processes.

Last audited: September 2026. Re-run before any release; upstream projects do
relicense.

---

## Why Apache-2.0 and not MIT

Apache-2.0 grants patent rights explicitly (§3). MIT does not address patents at
all. In a drug discovery repository, contributors are disproportionately likely
to work somewhere with a patent portfolio, and a downstream user is
disproportionately likely to care whether contributing to the project implies a
patent licence. Apache-2.0 answers that question in the text instead of leaving
it to be litigated.

The cost is a slightly longer licence file and a `NOTICE` obligation. That is a
good trade.

---

## Imported libraries

These are `import`ed into the process. Their licences apply to this codebase.

| Library | Licence | SPDX | Verified |
|---|---|---|---|
| RDKit | BSD 3-Clause | `BSD-3-Clause` | GitHub API |
| NumPy | BSD 3-Clause | `BSD-3-Clause` | upstream `LICENSE.txt` |
| pandas | BSD 3-Clause | `BSD-3-Clause` | upstream `LICENSE` |
| scikit-learn | BSD 3-Clause | `BSD-3-Clause` | upstream `COPYING` |
| requests | Apache-2.0 | `Apache-2.0` | upstream `LICENSE` |
| typer | MIT | `MIT` | upstream `LICENSE` |
| rich | MIT | `MIT` | upstream `LICENSE` |
| pydantic | MIT | `MIT` | upstream `LICENSE` |
| mcp (optional) | MIT | `MIT` | upstream `LICENSE` |

All permissive, all compatible with Apache-2.0 distribution. No copyleft in this
column, and adding any would change the licence of the whole project.

---

## External programs

These are invoked as **separate processes over files**. They are never imported
and never linked.

| Program | Source licence | Binary caveat | Used for |
|---|---|---|---|
| smina | Apache-2.0 | conda-forge build **links Open Babel (GPL-2.0)** | docking |
| AutoDock Vina 1.2+ | Apache-2.0 | — | docking |
| gnina | Apache-2.0 | links Open Babel in most builds | docking + CNN rescoring |
| fpocket | MIT | — | binding-site detection |
| Open Babel | **GPL-2.0** | — | format conversion, optional and never required |

### Why the process boundary matters

The GPL's copyleft attaches to *derivative works*. Linking a GPL library into a
program, or importing it into a running process, is generally understood to
create one. Invoking a separate executable and exchanging data through files is
generally understood not to — this is the same boundary that lets proprietary
software call `grep`.

That distinction is load-bearing here because of one specific fact:
**conda-forge's smina package depends on Open Babel.** You can confirm it
yourself:

```
$ micromamba search -c conda-forge smina
 License         Apache-2.0
 Dependencies:
  - libboost >=1.82.0,<1.83.0a0
  - libcxx >=18
  - openbabel
```

So while smina's *source* is Apache-2.0, the *binary you install* is linked
against a GPL-2.0 library and is therefore itself effectively GPL-encumbered as
distributed. Had `chilecule` used a Python binding that loaded that binary into
its own address space, the argument that this project is Apache-2.0 would be
considerably weaker.

Instead:

- `chilecule.tools.docking` builds an argument list and calls
  `subprocess.run(["smina", ...])`.
- Data crosses the boundary as SDF and PDB files on disk.
- No smina, Vina, gnina, fpocket, or Open Babel code is imported, linked,
  vendored, or redistributed.
- The installer fetches these from conda-forge at install time; this repository
  ships none of them.

If you disagree with this analysis, the practical consequence is contained: run
`chilecule` without the `dock` tier and the four workflows still function, minus
docking and pocket detection.

### What is deliberately not used

**Open Babel's Python bindings (`openbabel`, `pybel`).** These are the standard
way tutorials do format conversion, and importing them would make this codebase
a GPL derivative. RDKit covers essentially every conversion this project needs.
Open Babel remains available as an optional `obabel` subprocess, and nothing
requires it.

**Meeko (LGPL-2.1).** The Forli lab's ligand preparation tool for Vina. LGPL
permits use by an unmodified-library-linking program, but the boundary for a
Python `import` is contested, and some legal teams reject it outright. Where
Meeko is wanted, invoke `mk_prepare_ligand.py` as a subprocess, which is how it
is normally used anyway.

**MDAnalysis (LGPLv2.1+/LGPLv3+) and OpenMM (mixed MIT and LGPL).** Both are
excellent and neither is needed for any shipped workflow. If molecular dynamics
is ever added, it belongs behind the same process boundary or in a separately
licensed optional package.

---

## Data sources

Data licences are frequently more restrictive than software licences and are
routinely ignored. They are not ignored here.

| Source | Licence | Consequence for this project |
|---|---|---|
| ChEMBL | **CC BY-SA 3.0** | Attribution with release version; **share-alike on derivatives** |
| RCSB PDB | CC0 1.0 | No restriction |
| PDBe (SIFTS mappings) | CC0 / EMBL-EBI terms | No restriction |
| UniProt | CC BY 4.0 | Attribution |
| PubChem | Public domain | No restriction |
| DrugBank | CC BY-NC 4.0 | **Excluded — non-commercial** |

### ChEMBL share-alike

ChEMBL is CC BY-SA 3.0. Redistributing a *derived* dataset would oblige this
project to license that dataset under CC BY-SA 3.0 as well, which conflicts
awkwardly with an Apache-2.0 repository and creates an obligation for every
downstream user.

The design response is simple: **ChEMBL is queried at runtime and no derived
dataset is committed to this repository.** Responses are cached under
`~/.cache/chilecule/`, outside the source tree and outside version control.

Attribution travels with the data. `ChemblClient.release()` reports the version,
`CurationReport` carries it, every curated DataFrame stores it in
`frame.attrs["chembl_release"]`, and every generated report prints it under
Provenance:

```
- chembl_release: ChEMBL_37
- chembl_license: CC BY-SA 3.0
```

One further ChEMBL caveat, from their own licensing page: some computed property
columns are derived from commercial software, and users should not extract those
in isolation to train models intended to replicate the commercial calculation.
This project computes its own descriptors with RDKit and does not use ChEMBL's
property columns.

### DrugBank

DrugBank is CC BY-NC. Any pipeline that touches it cannot be used commercially,
which would quietly poison every downstream user of an Apache-2.0 library. It is
not wired in and should not be added.

---

## Rules for contributors

1. **No new import without a licence check.** Record it in the table above.
2. **No GPL or LGPL library imported into the core.** If a tool is only
   available under copyleft, wrap it as a subprocess and document the boundary
   here.
3. **No derived dataset committed to the repository**, especially not one
   derived from ChEMBL.
4. **Data licences count.** Check the terms of any new data source and record
   them, including any non-commercial clause.

---

## Verifying this yourself

```bash
# Licences of the installed Python environment
pip install pip-licenses && pip-licenses --format=markdown

# Upstream licence of any GitHub dependency
curl -s https://api.github.com/repos/rdkit/rdkit | python3 -c \
  "import sys,json; print(json.load(sys.stdin)['license']['spdx_id'])"

# Confirm the conda-forge smina -> Open Babel dependency
micromamba search -c conda-forge smina
```
