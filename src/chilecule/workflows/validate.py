"""Retrospective validation: does the cascade actually enrich for actives?

This is the workflow that makes the rest of the project answerable. Everything
else produces a ranked list; this measures whether such a list would have found
known actives hidden among plausible negatives.

The procedure:

1. Pull curated actives and confirmed inactives for a target from ChEMBL.
2. **Check the negative set before using it.** Train a descriptor-only
   classifier to separate actives from negatives. If it succeeds, the two sets
   differ in ways that have nothing to do with the target, and any enrichment
   measured against them is an artifact -- so the workflow reports the failure
   prominently rather than proceeding quietly.
3. Rank the pooled set with the method under test.
4. Report ROC-AUC, BEDROC, and enrichment factors, each against its ceiling.

Step 2 is the step that is almost always missing. Without it, a validation
reports a large enrichment factor, everyone is pleased, and the method fails
prospectively because it was separating molecular weight all along.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..bench.decoys import decoy_bias_report, property_matched_decoys
from ..bench.metrics import evaluate, max_enrichment_factor
from ..tools.chem import properties
from ..tools.chembl import ChemblClient, censored_inactives, curate_activities
from ..tools.sar import fingerprint
from ..tools.chem import parse_smiles
from .report import Report, Section, base_provenance

ACTIVE_THRESHOLD = 7.0  # pChEMBL >= 7 is 100 nM or better


def similarity_to_actives(query_smiles: str, reference_smiles: list[str]) -> float:
    """Maximum ECFP4 Tanimoto to a set of reference actives.

    The baseline every structure-based method must beat. Ligand-based
    similarity search is nearly free and works remarkably well, so a docking
    protocol that does not outperform it is not earning its runtime.
    """
    from rdkit import DataStructs

    query = parse_smiles(query_smiles)
    if query is None:
        return 0.0
    references = [fingerprint(m) for m in (parse_smiles(s) for s in reference_smiles) if m is not None]
    if not references:
        return 0.0
    return max(DataStructs.BulkTanimotoSimilarity(fingerprint(query), references))


def build(
    target: str,
    *,
    max_activity_records: int = 8000,
    active_threshold: float = ACTIVE_THRESHOLD,
    n_reference_actives: int = 25,
    target_active_fraction: float = 0.02,
    n_replicates: int = 20,
    client: ChemblClient | None = None,
) -> Report:
    """Validate a ligand-based ranking on a target's own experimental data."""
    client = client or ChemblClient()
    report = Report(workflow="Retrospective validation", subject=target)

    targets = client.find_targets(target)
    if targets.empty:
        report.warn(f"No ChEMBL target matched {target!r}.")
        return report

    chembl_target = targets.iloc[0]
    release = client.release()
    raw = client.fetch_activities(
        chembl_target["target_chembl_id"], max_records=max_activity_records
    )
    curated, curation = curate_activities(
        raw, target_chembl_id=chembl_target["target_chembl_id"], chembl_release=release
    )
    inactives = censored_inactives(raw)

    actives = curated[curated["pchembl"] >= active_threshold]
    if len(actives) < 20 or len(inactives) < 20:
        report.warn(
            f"Insufficient data for validation: {len(actives)} actives at pChEMBL >= "
            f"{active_threshold} and {len(inactives)} confirmed inactives. "
            "At least 20 of each are needed."
        )
        report.provenance = base_provenance(target_query=target, chembl_release=release)
        return report

    report.add(
        Section(
            title="Evaluation set",
            body=(
                f"{len(actives)} actives (pChEMBL >= {active_threshold}, i.e. "
                f"{10 ** (9 - active_threshold):.0f} nM or better) and {len(inactives)} "
                "compounds with right-censored inactive measurements against the same "
                f"target.\n\n```\n{curation.summary()}\n```"
            ),
            data={"n_actives": len(actives), "n_inactives": len(inactives)},
        )
    )

    # --------------------------------------------- Negative-set bias check
    bias = decoy_bias_report(actives, inactives)
    report.add(
        Section(
            title="Negative-set bias check",
            body=(
                f"```\n{bias.summary()}\n```\n\n"
                "A descriptor-only classifier was trained to tell actives from negatives "
                "using physicochemical properties alone -- no structural fingerprints, no "
                "target information. If it succeeds, the two groups are distinguishable for "
                "reasons unrelated to binding, and enrichment measured on this pair reflects "
                "that difference rather than the method under test."
            ),
            data={"bias": bias.to_dict()},
        )
    )

    if bias.verdict == "FAIL":
        report.warn(
            f"The negative set is biased (descriptor-only AUC {bias.auc:.2f}). Enrichment "
            "figures below are reported for transparency but do NOT demonstrate that the "
            "method works -- a classifier can reach them without any knowledge of the "
            "target. This is the expected outcome for ChEMBL confirmed inactives on a "
            "well-studied target, where actives and inactives come from different papers "
            "pursuing different chemotypes."
        )

    # ------------------------------------------------------ Ranking method
    pool = pd.concat(
        [
            actives[["smiles"]].assign(label=1),
            inactives[["smiles"]].assign(label=0),
        ],
        ignore_index=True,
    ).drop_duplicates("smiles")

    reference = actives.head(n_reference_actives)["smiles"].tolist()
    held_out = pool[~pool["smiles"].isin(reference)].reset_index(drop=True)

    held_out["similarity_score"] = [
        similarity_to_actives(s, reference) for s in held_out["smiles"]
    ]
    profile = [properties(s) for s in held_out["smiles"]]
    held_out["qed_score"] = [p.qed if p else 0.0 for p in profile]

    # Subsample to a realistic active fraction.
    #
    # ChEMBL yields roughly as many actives as inactives for a well-studied
    # target, but a real screening library is around 0.1-1% active. Evaluating
    # on a 50%-active set is not merely optimistic, it is degenerate: the
    # theoretical EF ceiling at 1% collapses toward 1.0, so even a perfect
    # method cannot post an enrichment factor that looks like anything. Every
    # metric below is therefore computed on subsamples drawn at
    # ``target_active_fraction``, repeated ``n_replicates`` times, and reported
    # as mean +/- standard deviation -- one subsample would be a number with no
    # error bar attached to it.
    all_actives = held_out[held_out["label"] == 1].reset_index(drop=True)
    all_inactives = held_out[held_out["label"] == 0].reset_index(drop=True)
    n_actives_per_replicate = max(
        5, min(len(all_actives), int(round(len(all_inactives) * target_active_fraction
                                           / (1 - target_active_fraction))))
    )

    method_names = [
        f"ECFP4 similarity to {n_reference_actives} known actives",
        "QED (drug-likeness only)",
        "random ordering",
    ]
    collected: dict[str, list[dict[str, float]]] = {name: [] for name in method_names}

    for replicate in range(n_replicates):
        rng = np.random.default_rng(replicate)
        sampled = all_actives.sample(n=n_actives_per_replicate, random_state=replicate)
        subset = pd.concat([sampled, all_inactives], ignore_index=True)
        labels = subset["label"].to_numpy()

        scores_by_method = {
            method_names[0]: subset["similarity_score"].to_numpy(),
            method_names[1]: subset["qed_score"].to_numpy(),
            method_names[2]: rng.random(len(subset)),
        }
        for name, scores in scores_by_method.items():
            metrics = evaluate(scores, labels)
            collected[name].append(
                {
                    "ROC_AUC": metrics.roc_auc,
                    "BEDROC_a20": metrics.bedroc_alpha20,
                    **{f"EF_{k}": v for k, v in metrics.enrichment.items()},
                }
            )

    def _mean_std(values: list[float]) -> str:
        array = np.asarray(values, dtype=float)
        return f"{array.mean():.3f} ± {array.std():.3f}"

    metric_keys = list(collected[method_names[0]][0].keys())
    rows = [
        {"method": name, **{key: _mean_std([r[key] for r in runs]) for key in metric_keys}}
        for name, runs in collected.items()
    ]

    n_evaluated = n_actives_per_replicate + len(all_inactives)
    rows.append(
        {
            "method": "-- theoretical ceiling --",
            "ROC_AUC": "1.000",
            "BEDROC_a20": "1.000",
            **{
                f"EF_{int(f * 100)}%": f"{max_enrichment_factor(n_evaluated, n_actives_per_replicate, f):.3f}"
                for f in (0.01, 0.05, 0.10)
            },
        }
    )

    report.add(
        Section(
            title="Enrichment results",
            body=(
                f"Each replicate ranks {n_evaluated} compounds: "
                f"{n_actives_per_replicate} actives sampled from {len(all_actives)} available, "
                f"plus all {len(all_inactives)} confirmed inactives -- an active fraction of "
                f"{n_actives_per_replicate / n_evaluated:.1%}, chosen to resemble a real "
                f"screening library rather than the ~50% that ChEMBL yields directly. "
                f"Reported as mean ± SD over {n_replicates} replicates.\n\n"
                f"The {n_reference_actives} reference actives used to build the similarity "
                "query are excluded from every evaluation set -- scoring a method on the data "
                "it was given is not a validation.\n\n"
                "The random row is the floor and the ceiling row is the maximum any method "
                "could achieve on a set of this composition. An enrichment factor is "
                "uninterpretable without both."
            ),
            table=pd.DataFrame(rows),
            data={
                "n_evaluated_per_replicate": n_evaluated,
                "n_actives_per_replicate": n_actives_per_replicate,
                "active_fraction": n_actives_per_replicate / n_evaluated,
                "n_replicates": n_replicates,
                "raw": {name: runs for name, runs in collected.items()},
            },
        )
    )

    similarity_auc = float(np.mean([r["ROC_AUC"] for r in collected[method_names[0]]]))
    qed_auc = float(np.mean([r["ROC_AUC"] for r in collected[method_names[1]]]))
    report.add(
        Section(
            title="Interpretation",
            body=(
                f"Similarity search reaches ROC-AUC {similarity_auc:.3f}; drug-likeness "
                f"alone reaches {qed_auc:.3f}; random is 0.5 by construction.\n\n"
                "Two conclusions follow, and both matter more than the headline number.\n\n"
                "First, **the QED row is a control, not a method.** QED knows nothing about "
                f"the target. Whatever it scores above 0.5 is the amount of apparent "
                "performance available for free from the composition of the evaluation set. "
                "Any real method must be judged against that number, not against 0.5.\n\n"
                "Second, **similarity search is the bar for structure-based methods.** It "
                "costs milliseconds and requires no protein structure. A docking protocol "
                "that does not clearly beat this row is not paying for its runtime, however "
                "physically satisfying its poses look."
            ),
            data={"similarity_auc": similarity_auc, "qed_auc": qed_auc},
        )
    )

    report.provenance = base_provenance(
        target_query=target,
        chembl_target=chembl_target["target_chembl_id"],
        chembl_release=release,
        chembl_license="CC BY-SA 3.0",
        active_threshold=active_threshold,
        n_actives=len(actives),
        n_inactives=len(inactives),
        negative_set_verdict=bias.verdict,
    )
    return report
