"""What is my series telling me?

Takes a congeneric series with activity data and produces the analysis a
project team would ask a computational chemist for: which position does what,
whether the SAR is additive, where the cliffs are, and -- the question that
gets asked too late -- whether potency is being bought with lipophilicity.

Input is a CSV or DataFrame with a ``smiles`` column and an activity column.
No target lookup, no network, no docking. This is the workflow for data you
already have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..tools.chem import lipophilic_efficiency, properties
from ..tools.rgroup import decompose, free_wilson, substituent_table
from ..tools.sar import activity_cliffs, scaffold_summary
from .report import Report, Section, base_provenance


def build(
    df: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    activity_col: str = "pchembl",
    id_col: str = "id",
    core: str | None = None,
    subject: str = "series",
) -> Report:
    """Analyse a congeneric series that already has activity data."""
    report = Report(workflow="Series analysis", subject=subject)

    if smiles_col not in df or activity_col not in df:
        report.warn(
            f"Input needs a {smiles_col!r} column and an {activity_col!r} column; "
            f"found {list(df.columns)}."
        )
        return report

    work = df.dropna(subset=[smiles_col, activity_col]).reset_index(drop=True)
    if id_col not in work:
        work[id_col] = [f"CPD_{i + 1:03d}" for i in range(len(work))]

    if len(work) < 5:
        report.warn(f"Only {len(work)} compounds with activity. Nothing below will be stable.")
        return report

    activities = work[activity_col].astype(float)
    report.add(
        Section(
            title="Dataset",
            body=(
                f"{len(work)} compounds spanning {activities.min():.2f} to "
                f"{activities.max():.2f} {activity_col} "
                f"({activities.max() - activities.min():.1f} log units).\n\n"
                + (
                    "A range under 2 log units is a narrow window in which to detect SAR; "
                    "much of what follows may be assay noise.\n\n"
                    if activities.max() - activities.min() < 2 else ""
                )
                + f"Median {activities.median():.2f}, "
                f"{int((activities >= 8).sum())} compounds at 10 nM or better."
            ),
            data={
                "n_compounds": len(work),
                "activity_range": round(float(activities.max() - activities.min()), 2),
            },
        )
    )

    # ------------------------------------------------------- Is it one series?
    scaffolds = scaffold_summary(work, smiles_col=smiles_col, activity_col=activity_col)
    if len(scaffolds) > 1:
        top_share = float(scaffolds.iloc[0]["fraction_of_set"])
        if top_share < 0.6:
            report.warn(
                f"This is not one congeneric series: the largest Bemis-Murcko scaffold "
                f"covers only {top_share:.0%} of the set across {len(scaffolds)} scaffolds. "
                "R-group and Free-Wilson analysis assume a shared core, so run them per "
                "series rather than over the whole table."
            )
        report.add(
            Section(
                title="Series composition",
                body=f"{len(scaffolds)} distinct Bemis-Murcko scaffolds.",
                table=scaffolds.head(8),
            )
        )

    # ---------------------------------------------------- R-group SAR
    decomposition = decompose(work, smiles_col=smiles_col, core=core, activity_col=activity_col)
    if decomposition is None:
        report.warn(
            "No common core could be found, so no R-group table. The compounds are too "
            "diverse for decomposition -- cluster them into series first."
        )
    else:
        table = substituent_table(decomposition, activity_col=activity_col)
        report.add(
            Section(
                title="R-group SAR",
                body=(
                    f"Common core found across {decomposition.n_decomposed} compounds "
                    f"({decomposition.n_failed} did not match), giving "
                    f"{len(decomposition.positions)} variable positions.\n\n"
                    "`spread` is the gap between the best and worst compound carrying that "
                    "substituent at that position. A large spread means the position is "
                    "sensitive to what else is on the molecule -- read those rows alongside "
                    "the Free-Wilson fit below rather than on their own."
                ),
                table=table,
                max_rows=25,
                data=decomposition.to_dict(),
            )
        )

        model = free_wilson(decomposition, activity_col=activity_col)
        if model is not None:
            report.add(
                Section(
                    title="Free-Wilson additivity",
                    body=(
                        f"{model.interpretation}\n\n"
                        f"Cross-validated R² = {model.r_squared:.2f} over {model.n_compounds} "
                        f"compounds and {model.n_terms} substituent terms. Cross-validated "
                        "rather than fitted: a one-hot model with a term per substituent can "
                        "memorize a small series completely, and the training fit would "
                        "always look excellent.\n\n"
                        "*Free-Wilson cannot extrapolate. A substituent absent from the "
                        "series has no coefficient, so this ranks what was made and is "
                        "silent on what was not.*"
                    ),
                    table=model.contributions.head(15) if not model.contributions.empty else None,
                    data=model.to_dict(),
                )
            )

    # ---------------------------------------------------- Cliffs
    cliffs = activity_cliffs(work, smiles_col=smiles_col, id_col=id_col, activity_col=activity_col)
    if not cliffs.empty:
        report.add(
            Section(
                title="Activity cliffs",
                body=(
                    f"{len(cliffs)} pairs at least 80% similar by ECFP4 yet differing by more "
                    "than one log unit, ranked by SALI.\n\n"
                    "These are the highest-information compounds in the set: a large potency "
                    "swing from a small change localizes a specific interaction. They also "
                    "bound what any QSAR model can achieve here -- a model cannot be both "
                    "smooth and correct across a cliff."
                ),
                table=cliffs.head(10),
                data={"n_cliffs": len(cliffs)},
            )
        )

    # ---------------------------------------------------- Efficiency trajectory
    efficiency = _efficiency_frame(work, smiles_col, activity_col)
    if not efficiency.empty:
        report.add(
            Section(
                title="Are you buying potency with lipophilicity?",
                body=_efficiency_text(efficiency),
                table=efficiency.sort_values("LLE", ascending=False).head(12),
                data={
                    "median_LE": round(float(efficiency["LE"].median()), 3),
                    "median_LLE": round(float(efficiency["LLE"].median()), 2),
                    "potency_clogp_correlation": _correlation(efficiency),
                },
            )
        )

    report.provenance = base_provenance(
        n_compounds=len(work),
        activity_column=activity_col,
        core=decomposition.core if decomposition else None,
    )
    return report


def _efficiency_frame(work: pd.DataFrame, smiles_col: str, activity_col: str) -> pd.DataFrame:
    rows = []
    for _, record in work.iterrows():
        props = properties(record[smiles_col])
        if props is None:
            continue
        pactivity = float(record[activity_col])
        potency_nm = 10 ** (9 - pactivity)
        rows.append(
            {
                "smiles": record[smiles_col],
                activity_col: round(pactivity, 2),
                "mw": props.mw,
                "clogp": props.clogp,
                "heavy_atoms": props.heavy_atoms,
                "LE": (
                    round(1.3717 * pactivity / props.heavy_atoms, 3)
                    if props.heavy_atoms else None
                ),
                "LLE": lipophilic_efficiency(potency_nm, props.clogp),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["LE", "LLE"])


def _correlation(efficiency: pd.DataFrame) -> float | None:
    ignored = {"smiles", "mw", "clogp", "heavy_atoms", "LE", "LLE"}
    activity_col = [c for c in efficiency.columns if c not in ignored]
    if not activity_col or len(efficiency) < 5:
        return None
    return round(float(np.corrcoef(efficiency[activity_col[0]], efficiency["clogp"])[0, 1]), 3)


def _efficiency_text(efficiency: pd.DataFrame) -> str:
    """The question that gets asked too late on most programs."""
    correlation = _correlation(efficiency)
    median_lle = float(efficiency["LLE"].median())
    median_le = float(efficiency["LE"].median())

    lines = [
        f"Median ligand efficiency **{median_le:.3f}** kcal/mol per heavy atom; "
        f"median lipophilic efficiency **{median_lle:.2f}**.",
        "",
        "LE is potency per atom -- it stops you rewarding a compound for being large. "
        "LLE (pActivity − cLogP) is potency net of greasiness. Above 5 is healthy.",
    ]

    if correlation is not None:
        lines += [
            "",
            "Correlation between potency and cLogP across the series: "
            f"**{correlation:+.2f}**.",
        ]
        if correlation >= 0.5:
            lines.append(
                "That is the signature of a series buying potency with lipophilicity. It "
                "works, right up until solubility, promiscuity and hERG arrive together in "
                "preclinical. The compounds worth pursuing are the ones above the LLE "
                "trend, not the ones at the top of the potency column."
            )
        elif correlation <= -0.2:
            lines.append(
                "Potency is rising as lipophilicity falls, which is the healthy direction "
                "and unusual. Whatever is driving it is worth understanding and repeating."
            )
        else:
            lines.append(
                "No strong relationship, so potency in this series is not simply tracking "
                "greasiness."
            )

    best_lle = efficiency.nlargest(1, "LLE").iloc[0]
    lines += [
        "",
        f"Best compound by LLE is `{best_lle['smiles'][:60]}` at LLE {best_lle['LLE']:.2f} "
        f"(cLogP {best_lle['clogp']}), which is a better starting point for optimization "
        "than the most potent compound if the two differ.",
    ]
    return "\n".join(lines)
