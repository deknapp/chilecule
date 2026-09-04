"""Does the ranking actually work -- and is the benchmark even fair?

Two questions, in that order. The second one gates the first: if actives and
negatives are separable from physicochemical descriptors alone, then enrichment
measured on that pair is an artifact, however impressive the number looks.
"""

from chilecule.bench.decoys import decoy_bias_report
from chilecule.tools.chembl import ChemblClient, censored_inactives, curate_activities
from chilecule.workflows.validate import build

TARGET = "EGFR"


def main() -> None:
    client = ChemblClient()
    raw = client.fetch_activities("CHEMBL203", max_records=6000)
    actives, _ = curate_activities(raw, target_chembl_id="CHEMBL203")
    inactives = censored_inactives(raw)

    print("=" * 70)
    print("STEP 1 -- is the negative set fair?")
    print("=" * 70)
    bias = decoy_bias_report(actives.head(400), inactives.head(400))
    print(bias.summary())

    print("\n" + "=" * 70)
    print("STEP 2 -- enrichment, read in light of step 1")
    print("=" * 70)
    report = build(TARGET, max_activity_records=6000)
    print(report.to_markdown())


if __name__ == "__main__":
    main()
