"""SAR analysis: what the existing data says about how to make the series better.

Takes a target, curates its bioactivity, and reports the three things a
medicinal chemist wants from a dataset they did not generate: which scaffolds
carry the activity, which substitutions moved potency and by how much, and
where the activity cliffs are.

No compute beyond RDKit. Runs in about a minute on a laptop.
"""

from __future__ import annotations

import pandas as pd

from ..tools.chem import ligand_efficiency, lipophilic_efficiency, properties
from ..tools.chembl import ChemblClient, curate_activities
from ..tools.sar import (
    activity_cliffs,
    matched_pairs,
    scaffold_summary,
    transformation_summary,
)
from .report import Report, Section, base_provenance


def build(
    target: str,
    *,
    max_activity_records: int = 5000,
    max_compounds_for_pairs: int = 800,
    min_transformation_occurrences: int = 4,
    client: ChemblClient | None = None,
) -> Report:
    """Analyse the structure-activity relationships in a target's known actives."""
    client = client or ChemblClient()
    report = Report(workflow="SAR analysis", subject=target)

    targets = client.find_targets(target)
    if targets.empty:
        report.warn(f"No ChEMBL target matched {target!r}.")
        return report

    chembl_target = targets.iloc[0]
    release = client.release()
    raw = client.fetch_activities(
        chembl_target["target_chembl_id"], max_records=max_activity_records
    )
    actives, curation = curate_activities(
        raw, target_chembl_id=chembl_target["target_chembl_id"], chembl_release=release
    )

    if len(actives) < 30:
        report.warn(
            f"Only {len(actives)} compounds survived curation. SAR analysis on a set this "
            "small produces transformation statistics driven by single observations; "
            "treat everything below as anecdote, not trend."
        )
    if actives.empty:
        return report

    report.add(
        Section(
            title="Dataset",
            body=(
                f"Target **{chembl_target['target_chembl_id']}** "
                f"({chembl_target['pref_name']}).\n\n```\n{curation.summary()}\n```"
            ),
            data={"curation": curation.to_dict()},
        )
    )

    # ---------------------------------------------------------- Scaffolds
    scaffolds = scaffold_summary(actives)
    if not scaffolds.empty:
        top_share = float(scaffolds.iloc[0]["fraction_of_set"])
        report.add(
            Section(
                title="Scaffold distribution",
                body=(
                    f"{len(scaffolds)} Bemis-Murcko scaffolds across {len(actives)} compounds; "
                    f"the largest covers {top_share:.0%} of the set.\n\n"
                    "`activity_range` is the span between the weakest and most potent member "
                    "of a scaffold. A wide range on a well-populated scaffold means the "
                    "series is responsive to substitution and is worth optimizing; a narrow "
                    "range means substitution has not been productive there."
                ),
                table=scaffolds.head(12),
                data={"n_scaffolds": len(scaffolds), "top_scaffold_share": top_share},
            )
        )

    # ------------------------------------------------- Matched molecular pairs
    subset = actives.head(max_compounds_for_pairs)
    pairs = matched_pairs(subset)
    transformations = transformation_summary(pairs, min_occurrences=min_transformation_occurrences)

    if not transformations.empty:
        gains = transformations.head(10)
        losses = transformations.tail(10).iloc[::-1]
        report.add(
            Section(
                title="Transformations that gained potency",
                body=(
                    f"{len(pairs)} matched pairs over {len(subset)} compounds, reduced to "
                    f"{len(transformations)} transformations seen at least "
                    f"{min_transformation_occurrences} times.\n\n"
                    "`median_delta` is the change in pChEMBL when the transformation is "
                    "applied, in log units: +1.0 is a tenfold potency gain. Read "
                    "`std_delta` alongside it -- a large median with a large spread means "
                    "the transformation is context-dependent, and applying it blindly to a "
                    "new position will not reproduce the gain.\n\n"
                    "Near-duplicate rows are expected. Single-cut fragmentation captures the "
                    "same chemical change at several bond positions along a chain, so one "
                    "real transformation can appear as two or three entries differing only "
                    "in how much of the linker went with the core. Read them as one finding."
                ),
                table=gains,
                data={"n_matched_pairs": len(pairs), "n_transformations": len(transformations)},
            )
        )
        report.add(
            Section(
                title="Transformations that lost potency",
                body=(
                    "The mirror image of the table above -- each entry is the reverse "
                    "transformation. Useful as a list of changes already shown not to work, "
                    "which is information a design campaign otherwise rediscovers by "
                    "making the compounds again."
                ),
                table=losses,
            )
        )
    else:
        report.warn(
            "No transformation was observed often enough to report. The dataset is either "
            "too small or too structurally diverse for matched-pair analysis."
        )

    # ----------------------------------------------------- Activity cliffs
    cliffs = activity_cliffs(subset)
    if not cliffs.empty:
        report.add(
            Section(
                title="Activity cliffs",
                body=(
                    f"{len(cliffs)} pairs are at least 80% similar by ECFP4 yet differ by "
                    "more than one log unit in potency, ranked by SALI.\n\n"
                    "These pairs carry the most information per compound in the dataset: a "
                    "large potency swing from a small structural change localizes a specific "
                    "interaction. They also bound what any QSAR model can achieve here, "
                    "since a model cannot be both smooth and correct across a cliff. Report "
                    "this count next to any model metric."
                ),
                table=cliffs.head(12),
                data={"n_cliffs": len(cliffs), "max_sali": float(cliffs["sali"].max())},
            )
        )
    else:
        report.add(
            Section(
                title="Activity cliffs",
                body=(
                    "No activity cliffs detected at the default thresholds (ECFP4 similarity "
                    ">= 0.8, potency difference >= 1 log unit). The landscape is smooth, "
                    "which is favourable for similarity-based modelling."
                ),
            )
        )

    # -------------------------------------------------------- Efficiency
    efficiency_rows = []
    for _, row in actives.head(200).iterrows():
        props = properties(row["smiles"])
        if props is None:
            continue
        efficiency_rows.append(
            {
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "pchembl": row["pchembl"],
                "heavy_atoms": props.heavy_atoms,
                "clogp": props.clogp,
                "LE": ligand_efficiency(row["value_nm"], props.heavy_atoms),
                "LLE": lipophilic_efficiency(row["value_nm"], props.clogp),
            }
        )

    if efficiency_rows:
        efficiency = pd.DataFrame(efficiency_rows).dropna(subset=["LE"])
        report.add(
            Section(
                title="Ligand efficiency leaders",
                body=(
                    "Ranked by ligand efficiency rather than raw potency, which is how a "
                    "project picks a starting point rather than a finished compound. A "
                    "moderately potent, efficient, small compound has room to grow; a "
                    "potent, inefficient, large one usually does not.\n\n"
                    "LLE (pIC50 - cLogP) tracks whether potency is being bought with "
                    "lipophilicity. Above 5 is healthy; a series whose LLE falls as potency "
                    "rises is heading for solubility and promiscuity problems."
                ),
                table=efficiency.sort_values("LE", ascending=False).head(12),
                data={
                    "median_LE": float(efficiency["LE"].median()),
                    "median_LLE": float(efficiency["LLE"].dropna().median())
                    if efficiency["LLE"].notna().any() else None,
                },
            )
        )

    report.provenance = base_provenance(
        target_query=target,
        chembl_target=chembl_target["target_chembl_id"],
        chembl_release=release,
        chembl_license="CC BY-SA 3.0",
        n_compounds=len(actives),
        n_matched_pairs=len(pairs),
    )
    return report
