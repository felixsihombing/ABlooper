from pymol import cmd

structures = list()

for i in range(100):
    structures.append('PRED'+str(i))

for structure in structures[1:]:
    cmd.align(structure, structures[0])

cmd.center(structures[0])

cmd.color('red', 'chain D')
cmd.color('green', 'not chain D')
