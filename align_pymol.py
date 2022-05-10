from pymol import cmd

structures = cmd.get_object_list('(all)')

for structure in structures[1:]:
    cmd.align(structure, structures[0])

cmd.center(structures[0])

cmd.color('red', 'chain D')
cmd.color('green', 'not chain D')
cmd.color('yellow', 'chain M')
