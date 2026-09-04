"""What is my series telling me?

Pulls a real congeneric series out of ChEMBL -- the 4-anilinoquinazoline EGFR
inhibitors -- and runs the analysis a project team would ask for: R-group SAR,
Free-Wilson additivity, activity cliffs, and whether potency is being bought
with lipophilicity.

Swap the ChEMBL pull for `pd.read_csv("my_series.csv")` to run it on your own
data; the workflow only needs a `smiles` column and an activity column.
"""

from chilecule.tools.chembl import ChemblClient, curate_activities
from chilecule.tools.sar import murcko_scaffold, scaffold_summary
from chilecule.workflows.series import build


def main() -> None:
    client = ChemblClient()
    raw = client.fetch_activities("CHEMBL203", max_records=4000)
    actives, _ = curate_activities(raw, target_chembl_id="CHEMBL203")

    # Keep one congeneric series: R-group analysis assumes a shared core.
    dominant = scaffold_summary(actives).iloc[0]["scaffold"]
    series = (
        actives[actives["smiles"].map(murcko_scaffold) == dominant]
        .head(120)
        .rename(columns={"molecule_chembl_id": "id"})
    )
    print(f"{len(series)} compounds on the dominant scaffold\n")

    report = build(series, subject="EGFR 4-anilinoquinazolines")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
