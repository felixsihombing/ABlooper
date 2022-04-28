def to_pdb_line(atom_id, atom_type, amino_type, chain_ID, residue_id, coords):
    """Puts all the required info into a .pdb format
    """
    x, y, z = coords
    insertion = "$"
    if type(residue_id) is str:
        if residue_id[-1].isalpha():
            insertion = residue_id[-1]
            residue_id = int(residue_id[:-1])
        else:
            residue_id = int(residue_id)
    line = "ATOM  {:5d}  {:3s} {:3s} {:1s} {:3d}{:2s}  {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           {}  \n"
    line = line.format(atom_id, atom_type, amino_type, chain_ID, residue_id, insertion, x, y, z, atom_type[0])

    return line.replace("$", " ")