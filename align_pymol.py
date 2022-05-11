from pymol import cmd

structures = cmd.get_object_list('(all)')

for i in range(0, len(structures), 3):
    cmd.color('red', structures[i])
    if i != 0:
        cmd.align(structures[i], structures[0]) # align different pdbs
    try:
        cmd.align(structures[i+1], structures[i]) # algin different model of same pdb with eachother
        cmd.color('green', structures[i+1])
    except Exception:
        pass

    try:
        cmd.align(structures[i+2], structures[i])
        cmd.color('yellow', structures[i+2])
    except Exception:
        pass

cmd.center(structures[0])

cmd.show_as('sticks', 'resi 27-38')
cmd.show_as('sticks', 'resi 56-65')
cmd.show_as('sticks', 'resi 105-117')
