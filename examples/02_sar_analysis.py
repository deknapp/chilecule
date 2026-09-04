"""What the existing data says about how to improve the series.

Matched molecular pairs isolate single structural changes and report their
effect on potency; activity cliffs mark where the SAR is steepest and where any
QSAR model will hit its accuracy ceiling.
"""

from chilecule.workflows.sar import build

TARGET = "EGFR"


def main() -> None:
    report = build(TARGET, max_activity_records=5000, max_compounds_for_pairs=600)
    print(report.to_markdown())
    print(f"\nSaved: {report.save('runs')['markdown']}")


if __name__ == "__main__":
    main()
