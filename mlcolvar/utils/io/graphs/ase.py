from warnings import warn

__all__ = ["create_pdb_from_xyz"]

def create_pdb_from_xyz(input_filename: str, output_filename: str) -> str:
    """
    Convert the first frame of an XYZ file into a PDB file using ASE.
    This pdb file can then serve as the topology for MDTraj.

    Parameters:
        input_filename: Path to the input .xyz file.
        output_filename: Path to the output .pdb file.

    Returns:
        The path to the generated PDB file.
    """
    # Import ASE 
    try:
        from ase.io import read, write
        from ase import Atoms
    except ImportError as e:
        raise ImportError("ASE is required for xyz to pdb conversion.", e)

    atoms: Atoms = read(input_filename, index=0)

    if (atoms.cell == 0).all():
        warn("A topology file was generated from the xyz trajectory file but no cell information were provided!")
    if not atoms.pbc.any():
        warn("A topology file was generated from the xyz trajectory file but no PBC information were provided!")
    elif not atoms.pbc.all():
        warn( f"Partial PBC are not supported! The provided input has pbc {atoms.pbc}")

    write(output_filename, atoms, format='proteindatabank')
    return output_filename
