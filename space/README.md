---
title: chilecule
emoji: 🌶️
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# chilecule

Agentic drug discovery on fully open-source tooling — RDKit, ChEMBL,
AutoDock Vina, fpocket.

This Space exposes the **deterministic tool layer**: profiling, structural
alerts, developability liabilities, synthetic accessibility, ChEMBL lookup, and
the retrospective validation harness. Everything computes on the molecule you
type. Nothing is precomputed.

**The validation harness is the point of the project.** It pulls curated
actives and confirmed inactives for a target from ChEMBL, *checks the negative
set for bias before using it*, then measures whether a ranking actually
enriches for the actives — reported against a random baseline, a
drug-likeness-only control, and the theoretical ceiling. That bias check is the
step most published validations omit, and it is why so many enrichment figures
fail to reproduce prospectively.

**What is deliberately not here.** The agent-driven workflows call a language
model and would spend the deployer's money on every visitor, so they stay on
the command line. Docking is installed in this image but a run takes minutes
and a shared container is the wrong place to start one.

Source: https://github.com/deknapp/chilecule
