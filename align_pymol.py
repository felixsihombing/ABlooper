from pymol import cmd

structures = list()

for i in range(100):
    structures.append('PRED'+str(i))

for structure in structures[1:]:
    cmd.align(structure, structures[0])

