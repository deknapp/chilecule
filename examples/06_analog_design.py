"""What should I make next?

Proposes analogs of a hit using only transformations medicinal chemists have
already made against this target, each carrying the evidence for what it did.

Note what the output does NOT do: it never claims a predicted potency. The
median delta is what the transformation did in the contexts where it was
observed. When the parent falls outside the chemical space that evidence came
from, the expected effect reads "no decision".
"""

from chilecule.workflows.analogs import build

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


def main() -> None:
    report = build(GEFITINIB, "EGFR", max_analogs=25)
    print(report.to_markdown())
    print(f"\nSaved: {report.save('runs')['markdown']}")


if __name__ == "__main__":
    main()
