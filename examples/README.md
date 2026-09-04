# Examples

Each script is runnable as-is and prints real output. Timings are from a
MacBook (M-series, 10 cores); nothing here needs a GPU.

| Script | What it shows | Needs | Time |
|---|---|---|---|
| `01_target_dossier.py` | What is known about a target, across UniProt / PDB / ChEMBL | network | ~30 s |
| `02_sar_analysis.py` | Matched pairs, activity cliffs, ligand efficiency | network | ~60 s |
| `03_docking_with_control.py` | Docking gated on a redocking control | network + smina | ~2 min |
| `04_validation.py` | Whether the ranking actually enriches, and whether the benchmark is fair | network | ~90 s |

```bash
source .venv/bin/activate
python examples/01_target_dossier.py
```

The equivalent CLI commands are `chilecule dossier`, `sar`, `triage`, and
`validate`.
