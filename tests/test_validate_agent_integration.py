"""The agent row in the retrospective validation must be earned, not granted.

These tests monkeypatch the ChEMBL layer -- curation is tested in
test_chembl_curation.py and is not the subject here. What is under test is the
part that decides whether the agent's number is comparable to the baselines':
that it is scored on the same pool, that omissions sink instead of vanishing,
and that the QED control keeps its own identity once the agent is inserted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from chilecule.workflows import validate as validate_workflow

# Small, real, and structurally varied enough that a descriptor model has
# something to chew on. Actives share an aromatic-acid motif; inactives do not.
ACTIVE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1O", "COc1ccccc1C(=O)O",
    "Cc1ccccc1C(=O)O", "OC(=O)c1ccc(Cl)cc1", "OC(=O)c1ccc(F)cc1",
    "OC(=O)c1ccc(C)cc1", "OC(=O)c1ccc(O)cc1", "OC(=O)c1ccc(N)cc1",
    "OC(=O)c1cccc(Cl)c1", "OC(=O)c1cccc(F)c1", "OC(=O)c1cccc(C)c1",
    "OC(=O)c1cccc(O)c1", "OC(=O)c1cccc(N)c1", "OC(=O)c1ccc(Br)cc1",
    "OC(=O)c1ccc(I)cc1", "OC(=O)c1ccc(CC)cc1", "OC(=O)c1ccc(OC)cc1",
    "OC(=O)c1ccc(CO)cc1", "OC(=O)c1ccc(CN)cc1", "OC(=O)c1ccc(CF)cc1",
    "OC(=O)c1ccc(CCl)cc1",
]
INACTIVE_SMILES = [
    "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC", "CCCCCCCCCCC", "CCCCCCCCCCCC",
    "CCCCCCCCO", "CCCCCCCCCO", "CCCCCCCCCCO", "CCCCCCCCN", "CCCCCCCCCN",
    "CCCCCCCCCCN", "CCCCCCCCBr", "CCCCCCCCCBr", "CCCCCCCCCl", "CCCCCCCCCCl",
    "CCCCCCCCF", "CCCCCCCCCF", "CCCCCCCCI", "CCCCCCCCCI", "CCCCCCCCCC(C)C",
    "CCCCCCCCC(C)C", "CCCCCCCC(C)C",
]


class FakeChemblClient:
    def release(self):
        return "CHEMBL_TEST"

    def find_targets(self, query, organism="Homo sapiens"):
        return pd.DataFrame([{"target_chembl_id": "CHEMBL_T1", "pref_name": query}])

    def fetch_activities(self, target_chembl_id, max_records=8000):
        return pd.DataFrame({"placeholder": [1]})


@pytest.fixture(autouse=True)
def fake_chembl(monkeypatch):
    """Replace curation with fixed actives/inactives frames."""
    actives = pd.DataFrame({"smiles": ACTIVE_SMILES, "pchembl": [8.0] * len(ACTIVE_SMILES)})
    inactives = pd.DataFrame({"smiles": INACTIVE_SMILES})

    class Curation:
        def summary(self):
            return "fake curation"

    monkeypatch.setattr(
        validate_workflow, "curate_activities", lambda raw, **kw: (actives, Curation())
    )
    monkeypatch.setattr(validate_workflow, "censored_inactives", lambda raw: inactives)


def run(agent_ranker=None, **kwargs):
    return validate_workflow.build(
        "TESTTARGET",
        client=FakeChemblClient(),
        n_reference_actives=5,
        n_replicates=2,
        agent_ranker=agent_ranker,
        **kwargs,
    )


def method_rows(report):
    """The method column of the enrichment table."""
    for section in report.sections:
        if section.title == "Enrichment results":
            return section.table["method"].tolist()
    raise AssertionError("no enrichment section in report")


def test_without_a_ranker_there_is_no_agent_row():
    assert not any("agent" in m for m in method_rows(run()))


def test_agent_row_appears_when_a_ranker_is_supplied():
    def ranker(candidates, reference):
        return {s: 1.0 for s in candidates}

    rows = method_rows(run(agent_ranker=ranker))
    assert validate_workflow.AGENT_METHOD in rows


def test_ranker_receives_the_same_reference_actives_as_the_baseline():
    seen = {}

    def ranker(candidates, reference):
        seen["candidates"] = list(candidates)
        seen["reference"] = list(reference)
        return {s: 1.0 for s in candidates}

    run(agent_ranker=ranker)

    assert len(seen["reference"]) == 5
    # The reference actives are excluded from what the agent is asked to score,
    # exactly as they are excluded from the similarity baseline's evaluation set.
    assert not set(seen["reference"]) & set(seen["candidates"])


def test_ranker_is_never_shown_labels():
    """The ranker signature carries SMILES only -- no label can reach it."""
    captured = {}

    def ranker(candidates, reference):
        captured["candidates"] = candidates
        return {s: 1.0 for s in candidates}

    run(agent_ranker=ranker)
    assert all(isinstance(c, str) for c in captured["candidates"])


def test_an_oracle_agent_scores_a_perfect_auc():
    """The metric path must be able to reward the agent, or the row is theatre.

    Note this fixture's actives and inactives are trivially separable, so the
    similarity baseline also reaches 1.0 here. That is fine: what is under test
    is that a correct agent ranking survives the replicate and metric path
    intact, not that it wins a contest the fixture cannot stage.
    """
    actives = set(ACTIVE_SMILES)

    def oracle(candidates, reference):
        return {s: (100.0 if s in actives else 0.0) for s in candidates}

    report = run(agent_ranker=oracle)
    assert _auc_for(report, validate_workflow.AGENT_METHOD) == pytest.approx(1.0, abs=0.01)


def test_an_inverted_agent_scores_a_failing_auc():
    """The path must also be able to punish it -- a row that can only go up is not a measurement."""
    actives = set(ACTIVE_SMILES)

    def inverted(candidates, reference):
        return {s: (0.0 if s in actives else 100.0) for s in candidates}

    report = run(agent_ranker=inverted)
    assert _auc_for(report, validate_workflow.AGENT_METHOD) < 0.1


def test_omitted_candidates_sink_rather_than_disappear():
    """An agent that scores only the actives it is sure of must not win by silence."""
    actives = set(ACTIVE_SMILES)

    def selective(candidates, reference):
        # Scores two actives it is confident about, and refuses everything else.
        confident = [c for c in candidates if c in actives][:2]
        return {s: 100.0 for s in confident}

    report = run(agent_ranker=selective)
    interpretation = next(s for s in report.sections if s.title == "Interpretation")

    assert interpretation.data["agent"]["n_scored"] == 2
    assert interpretation.data["agent"]["coverage"] < 1.0
    # Every unscored compound, active or not, is ranked below the two scored
    # ones -- so the agent cannot post a perfect AUC off two lucky picks.
    assert _auc_for(report, validate_workflow.AGENT_METHOD) < 1.0


def test_partial_coverage_is_warned_about():
    def selective(candidates, reference):
        return {list(candidates)[0]: 50.0}

    report = run(agent_ranker=selective)
    assert any("did not score" in w for w in report.warnings)


def test_qed_control_keeps_its_own_number_when_the_agent_is_added():
    """Regression: the interpretation once read QED's AUC off a positional index."""
    baseline = run()
    with_agent = run(agent_ranker=lambda c, r: {s: 1.0 for s in c})

    def qed(report):
        section = next(s for s in report.sections if s.title == "Interpretation")
        return section.data["qed_auc"]

    assert qed(baseline) == pytest.approx(qed(with_agent))


def test_pool_cap_preserves_label_balance_and_is_disclosed():
    uncapped = run()
    capped = run(max_pool=20)

    def pool_size(report):
        section = next(s for s in report.sections if s.title == "Enrichment results")
        return section.data["n_evaluated_per_replicate"]

    assert pool_size(capped) < pool_size(uncapped)
    assert any("capped" in w for w in capped.warnings)


def _auc_for(report, method_name):
    section = next(s for s in report.sections if s.title == "Enrichment results")
    runs = section.data["raw"][method_name]
    return sum(r["ROC_AUC"] for r in runs) / len(runs)
