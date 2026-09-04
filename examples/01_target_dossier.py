"""Assemble what is publicly known about a target.

Reconciles UniProt, the PDB, and ChEMBL, which do not agree with each other
about what a target is, and judges whether the picture supports a
structure-based campaign.
"""

from chilecule.workflows.dossier import build

TARGET = "EGFR"


def main() -> None:
    report = build(TARGET, max_activity_records=4000)
    print(report.to_markdown())

    paths = report.save("runs")
    print(f"\nSaved: {paths['markdown']}  and  {paths['json']}")


if __name__ == "__main__":
    main()
