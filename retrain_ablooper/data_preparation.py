'''Functions for data preparation. Exctract CDR sequences and backbone coordinates 
from SabDab and reformat them to the model inputs'''

# from ABDB import database as db
import numpy as np
from rich.progress import track
import copy
import torch
from einops import rearrange

# torch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)

# dictionaries to convert one letter, three letter and numerical amino acid codes
aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
       "THR", "VAL", "TRP", "TYR", ]

short2long = {}
long2short = {}
short2num = {}

for ind in range(0, 20):
    long2short[aa3[ind]] = aa1[ind]
    short2long[aa1[ind]] = aa3[ind]
    short2num[aa1[ind]] = ind

# functions to filter entries in SAbDAb
def filter_abs(pdb_list):
    '''
    Filter a list of PDB ids obtained from SAbDab and removes FABS where one of the chains is missing or where
    heavy and light chains have the same name.
    '''
    filtered_list = []
    i = 0
    
    for pdb in track(pdb_list, description='Filter FABs'):
        i += 1
        fab = db.fetch(pdb).fabs[0]

        if fab.VH == fab.VL:
            continue
        elif fab.VH == 'NA' or fab.VL == 'NA':
            continue
        else:
            filtered_list.append(pdb)

    return filtered_list

# functions to extract and format relevant data from a SAbDab FAB. Given a FAB two dictionaries are returned for CDR and anchor 
# sequences and thier backbone coordinates
def split_structure_in_regions(fab):
    '''
    Split FAB into regions.

    Takes FAB as input an returns a dictionary with keys: regions, values: residues in region
    regions = ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4', 'fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4']
    '''
    ab_regions = dict()
    struc = fab.get_structure()

    for chain in [fab.VH, fab.VL]:

        # Chian.get_residues() is a generator that loops through residue
        for residue in struc[chain].get_residues():

            # residue.region indicates in which cdr or framework region the residue is
            if residue.region in ab_regions:
                ab_regions[residue.region].append(residue)
            else:
                ab_regions[residue.region] = [residue]

    return ab_regions

def get_slice(ab_regions, CDR):
    '''
    Returns a slice of residues containing a CDR plus two anchor residues on each side,
    given a FAB split in to regions and a spefied CDR.
    '''
    chain = CDR[0].lower()
    loop = CDR[1]

    slice = ab_regions['fw' + chain + loop][-2:]
    slice += ab_regions['cdr' + chain + loop]
    slice += ab_regions['fw' + chain + str(int(loop) + 1 )][:2]

    return slice

def cdr_anchor_seq(ab_regions, CDR):
    '''
    Retruns sequence of a CDR plus two anchors on each side,
    given a FAB split in to regions and a spefied CDR.
    '''
    slice = get_slice(ab_regions, CDR)
    CDR_seq = []

    for res in slice:
        CDR_seq.append(long2short[res.resname])

    return CDR_seq

def cdr_anchor_BB_coord(ab_regions, CDR):
    '''
    Returns coordinates of backbone atoms of a CDR plus two anchors on each side,
    given a FAB split in to regions and a spefied CDR.
    '''
    slice = get_slice(ab_regions, CDR)
    CDR_BB_coord = np.zeros((len(slice), 4, 3))
    BB_atoms = ["CA", "C", "N", "CB"]

    for i in range(len(slice)):
        res = slice[i]
        for j in range(len(BB_atoms)):
            atom = BB_atoms[j]

            # if residue is glycine use CA coordinates for CB
            if res.resname == 'GLY' and atom == 'CB':
                atom = "CA"
                
            coord = res[atom].coord

            CDR_BB_coord[i, j, :] = coord
    
    return CDR_BB_coord

def get_cdr_anchor_seqs(ab_regions, CDRs = ["H1", "H2", "H3", "L1", "L2", "L3"]):
    '''
    Get sequences of all CDRs in a FAB.

    Returns a dictionary with keys: CDRs, values: CDR + anchor sequence
    '''
    CDR_seqs = dict()

    for CDR in CDRs:
        CDR_seqs[CDR] = cdr_anchor_seq(ab_regions, CDR)
   
    return CDR_seqs

def get_cdr_anchor_BB_coords(cdr_residues, CDRs = ["H1", "H2", "H3", "L1", "L2", "L3"]):
    '''
    Get backbone coordinates of all CDRs in a FAB.
    
    Returns a dictionary with keys: CDRs, values: CDR + anchor backbone coordinates
    '''
    CDR_BB_coords = dict()

    for CDR in CDRs:
        CDR_BB_coords[CDR] = cdr_anchor_BB_coord(cdr_residues, CDR)
   
    return CDR_BB_coords

# function to retrieve PDB ids from SAbDab and runs functions in above cells for each individual FAB.
def get_sabdab_fabs(pdb_list):
    '''
    Get fabs from sabdab given a list of pdbs, extracts CDR sequences and coordinates and formats the data for the next steps.

    returns CDR_seqs: list of dictionaries,
                      each dictionary contains data of one FAB, keys: CDR, value: CDR sequence
    returns CDR_BB_coords: list of dictionaries,
                           each dictionary contains data of one FAB, keys: CDR, value: CDR backbone coordinates
    '''
    pdb_list = filter_abs(pdb_list)

    CDR_seqs = list()
    CDR_BB_coords = list()

    for pdb_id in track(pdb_list, description='Load data from SAbDab'):
        pdb = db.fetch(pdb_id)
        for fab in pdb.fabs:
            try: # some fab have errors and throw exceptions, ignore these
                ab_regions = split_structure_in_regions(fab)
                cdr_seqs = get_cdr_anchor_seqs(ab_regions)
                cdr_BB_coords = get_cdr_anchor_BB_coords(ab_regions)

                CDR_seqs.append(cdr_seqs)
                CDR_BB_coords.append(cdr_BB_coords)
            except Exception:
                pass

    return CDR_seqs, CDR_BB_coords

# functions that convert data extracted from SAbDab to model input
def encode(x, classes):
    '''
    One hot encodes a scalar x into a vector of length classes.
    This is the function used for Sequence encoding.
    '''
    one_hot = np.zeros(classes)
    one_hot[x] = 1

    return one_hot

def one_hot(num_list, classes=20):
    '''
    One hot encodes a 1D vector x.
    This is the function used for Sequence encoding.
    '''
    end_shape = (len(num_list), classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        finish[i] = encode(num_list[i], classes)

    return finish

def which_loop(loop_seq, cdr):
    '''
    Adds a one-hot encoded vector to each node describing which CDR it belongs to.
    '''
    CDRs = ["H1", "H2", "H3", "L1", "L2", "L3", "Anchor"]
    loop = np.zeros((len(loop_seq), len(CDRs)))
    loop[:, -1] = 1
    loop[2:-2] = np.array([1.0 if cdr == x else 0.0 for x in (CDRs)])[None].repeat(len(loop_seq) - 4, axis=0)

    return loop

def positional_encoding(sequence, n=5):
    '''
    Gives the network information on how close each resdiue is to the anchors
    '''
    encs = []
    L = len(sequence)
    for i in range(n):
        encs.append(np.cos((2 ** i) * np.pi * np.arange(L) / L))
        encs.append(np.sin((2 ** i) * np.pi * np.arange(L) / L))

    return np.array(encs).transpose()

def res_to_atom(res_encoding, n_atoms=4):
    '''
    Adds a one-hot encoded vector to each node describing what atom type it is.
    '''
    out_shape = (res_encoding.shape[0], n_atoms, 41)
    atom_encoding = np.zeros(out_shape)

    for i in range(len(res_encoding)):
        for j in range(n_atoms):
            atom_encoding[i, j, 0:37] = res_encoding[i]
            # add one-hot encoding for atom type
            atom_encoding[i, j, 37:] = one_hot([j], classes=n_atoms) 

    return atom_encoding

def prepare_input_loop(CDR_coord, CDR_seq, CDR):
    '''
    Generates input features to be fed into the network for a single CDR
    '''
    CDR_input_coord = copy.deepcopy(CDR_coord)
    # put CDR residues equally spaced on straight line between anchor residues 
    CDR_input_coord[1:-1] = np.linspace(CDR_coord[1], CDR_coord[-2], len(CDR_coord) - 2)
    # CDR_input_coord = rearrange(torch.tensor(CDR_input_coords), "i a d -> () (i a) d").float()

    one_hot_encoding = one_hot(np.array([short2num[amino] for amino in CDR_seq]))
    loop = which_loop(CDR_seq, CDR)
    positional = positional_encoding(CDR_seq)
    res_encoding = np.concatenate([one_hot_encoding, positional, loop], axis=1)
    atom_encoding = res_to_atom(res_encoding)

    # encoding = res_to_atom(torch.tensor(np.concatenate([one_hot_encoding, positional, loop], axis=1)).float())
    # encoding = rearrange(encoding, "i a d -> () (i a) d")

    return CDR_input_coord, atom_encoding

def prepare_model_input(CDR_seq, CDR_BB_coord):
    '''
    Prepares model inputs for a single FAB
    '''
    encodings = []
    geomins = []
    
    for CDR in CDR_BB_coord:
        geom, encode = prepare_input_loop(CDR_BB_coord[CDR], CDR_seq[CDR], CDR)
        encodings.append(encode)
        geomins.append(geom)

    # concatenate encodings and geoms into single array
    encodings = np.concatenate(encodings, axis=0)
    geomins = np.concatenate(geomins, axis=0)
    # format to tensor
    encodings = torch.from_numpy(encodings)
    geomins = torch.from_numpy(geomins)
    # rearrange tensors that atoms in one residue are nolonger grouped
    encodings = rearrange(encodings, "i a d -> (i a) d")
    geomins = rearrange(geomins, "i a d -> (i a) d")

    return geomins, encodings

def prepare_model_inputs(CDR_seqs, CDR_BB_coords):
    '''
    Prepares model inputs for a list of FABs
    '''
    encodings = []
    geomins = []

    for i in track(range(len(CDR_seqs)), description='Preparing model inputs'):
        geom, encode = prepare_model_input(CDR_seqs[i], CDR_BB_coords[i])
        encodings.append(encode)
        geomins.append(geom)

    return geomins, encodings

def prepare_model_output(CDR_BB_coords):
    '''
    Prepares model outputs for training, formated identically to inputs
    '''
    geomouts = []
    for CDR_BB_coord in track(CDR_BB_coords, description='Preparing model outputs'):
        geomout = []
        for _, coords in CDR_BB_coord.items():
            geomout.append(coords)

        # concatenate geoms into single array
        geomout = np.concatenate(geomout, axis=0)
        # format to tensor
        geomout = torch.from_numpy(geomout)
        # rearrange tensor
        geomout = rearrange(geomout, "i a d -> (i a) d")

        geomouts.append(geomout)
    return geomouts

def concatenate_data(encodings, geomins, geomouts, masks):
    '''
    Puts encodings, geomins and geomouts into a single array.
    '''
    data = []
    for i in range(len(encodings)):
        # potentially change list to dict
        data.append({'encodings': encodings[i],
                     'geomins': geomins[i],
                     'geomouts': geomouts[i],
                     'mask': masks[i]})

    return data

# creat mask
def create_mask(node_encodings, mask_lenght=504):
    '''
    Function that creates a mask for each element in a array. Mask is of length <mask_lenght> 
    with 1s where there is an atom and filled up with 0s
    '''
    masks = []
    for node_encoding in node_encodings:
        mask = torch.zeros((mask_lenght))
        mask[:len(node_encoding)] = 1
        masks.append(mask)

    return masks
    
# padding
def pad_tensor(tensor, out_lenght=504):
    '''
    Pads all tensors with 0 to the same number of nodes
    '''
    ndim = tensor.shape[1]
    outtensor = torch.zeros((out_lenght, ndim))
    outtensor[:tensor.shape[0], :] = tensor
    return outtensor

def pad_list_of_tensors(tensors, pad_length=504):
    '''
    Pads all tensors in a list
    '''
    tensors_padded = []

    for i in range(len(tensors)):
        tensors_padded.append(pad_tensor(tensors[i], out_lenght=pad_length))

    return tensors_padded
