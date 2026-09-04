"""Docking, gated on a control that proves the protocol works.

Demonstrates the finding that shaped this project's receptor preparation:
re-docking a nucleotide into a kinase site fails when Mg(2+) has been stripped
as a crystallization additive, and passes when it is retained.

Requires a docking program:  micromamba install -c conda-forge smina
"""

from pathlib import Path

from rdkit import Chem

from chilecule.tools.docking import dock, find_program, redock_control
from chilecule.tools.structure import (
    box_from_ligand,
    extract_ligand_mol,
    extract_ligands,
    fetch_pdb,
    prepare_receptor,
)

PDB_ID = "5CNN"          # EGFR kinase domain, 1.9 A, AMP-PNP bound
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"


def main() -> None:
    program = find_program()
    if program is None:
        raise SystemExit(
            "No docking program on PATH.\n"
            "  micromamba install -c conda-forge smina"
        )
    print(f"Docking program: {program}\n")

    structure = fetch_pdb(PDB_ID)
    ligands = extract_ligands(structure)
    print(f"{PDB_ID} heteroatom groups:")
    for lig in ligands[:6]:
        print(f"  {lig.residue_name:>4}  {lig.category:<9} {lig.n_atoms:>3} atoms")

    native = next(lig for lig in ligands if lig.category == "ligand")
    box = box_from_ligand(structure, native)
    print(f"\nSite from {native.residue_name}: {box.to_dict()}")

    # Write the crystallographic ligand out with correct bond orders, taken
    # from the PDB Chemical Component Dictionary. Without this the RMSD
    # comparison runs against a molecule with the right shape and the wrong
    # chemistry.
    mol = extract_ligand_mol(structure, native)
    native_sdf = Path(structure).with_name(f"{Path(structure).stem}_native.sdf")
    writer = Chem.SDWriter(str(native_sdf))
    writer.write(mol)
    writer.close()

    print("\n" + "=" * 70)
    print("REDOCKING CONTROL -- the same site, prepared two ways")
    print("=" * 70)

    for keep_metals, label in [(False, "Mg stripped as an additive"),
                               (True, "Mg retained as a catalytic metal")]:
        receptor = prepare_receptor(
            structure,
            keep_chain=native.chain_id,
            keep_metals=keep_metals,
            out_path=Path(structure).with_name(f"rec_metals_{keep_metals}.pdb"),
        )
        control = redock_control(receptor, native_sdf, box, exhaustiveness=16)
        verdict = "PASS" if control.passed else "FAIL"
        print(f"\n  {label}")
        print(f"    RMSD to crystallographic pose : {control.rmsd_display()}")
        print(f"    Verdict                        : {verdict}")

    print("\n" + "=" * 70)
    print("Docking erlotinib into the validated receptor")
    print("=" * 70)

    receptor = prepare_receptor(structure, keep_chain=native.chain_id, keep_metals=True)
    result = dock(receptor, ERLOTINIB, box, num_modes=5)
    print(f"\n  best score : {result.best_score} kcal/mol")
    print(f"  poses      : {len(result.poses)}")
    print(
        "\n  Note: this is a ranking signal, not a predicted affinity. Docking"
        "\n  scores correlate weakly with measured potency across diverse"
        "\n  chemistry, and cannot be compared between different targets."
    )


if __name__ == "__main__":
    main()
