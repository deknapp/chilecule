"""Profile a compound the way a chemist would ask about one.

The daily workhorse: what is it, is it drug-like, what liabilities does it
carry, can it be made, has anyone made it, and what else does it hit.
"""

from chilecule.workflows.profile import build

COMPOUNDS = [
    "CC(=O)Oc1ccccc1C(=O)O",                                    # aspirin
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",               # erlotinib
    "CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCCCN1CCCCC1",             # a novel analog
]


def main() -> None:
    report = build(COMPOUNDS)
    print(report.to_markdown())
    print(f"\nSaved: {report.save('runs')['markdown']}")


if __name__ == "__main__":
    main()
