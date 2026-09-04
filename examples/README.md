# Examples

Each script is runnable as-is and prints real output. Timings are from a
MacBook (M-series, 10 cores); nothing here needs a GPU.

## Start from a molecule

These are the ones a medicinal chemist on a live program would reach for.

| Script | What it shows | Needs | Time |
|---|---|---|---|
| `05_compound_profile.py` | Properties, alerts, synthesis, novelty, promiscuity | network | ~20 s |
| `06_analog_design.py` | What to make next, with matched-pair evidence | network | ~90 s |
| `07_series_analysis.py` | R-group SAR, Free-Wilson, cliffs, LE/LLE trends | network | ~60 s |

## Start from a target

| Script | What it shows | Needs | Time |
|---|---|---|---|
| `01_target_dossier.py` | UniProt / PDB / ChEMBL reconciled | network | ~30 s |
| `02_sar_analysis.py` | Matched pairs and activity cliffs | network | ~60 s |
| `03_docking_with_control.py` | Docking gated on a redocking control | network + smina | ~2 min |
| `04_validation.py` | Whether the ranking enriches, and whether the benchmark is fair | network | ~90 s |

```bash
source .venv/bin/activate
python examples/05_compound_profile.py
```

The equivalent CLI commands are `chilecule profile`, `analogs`, `series`,
`dossier`, `sar`, `triage` and `validate`.

## Agent

`evals/agent_eval.py` drives the tools through an actual Claude agent and checks
its behaviour. It makes real API calls and is not part of `pytest`:

```bash
export ANTHROPIC_API_KEY=...
python evals/agent_eval.py
```
