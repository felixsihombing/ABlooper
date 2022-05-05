# functions to predict on val set and calculate RMSDs
from retrain_ablooper import *
import torch
import math

def predict_on_val_set(val_dataloader, model):
    '''
    Predicts structures of validation set
    '''
    preds = []
    node_features = []
    geomouts = []
    ids = []

    with torch.no_grad():
        model.eval()
        for data in track(val_dataloader, description='Predict validation set'):
            coordinates, geomout, node_feature, mask, id = data['geomins'].float().to(device), data['geomouts'].float().to(device), data['encodings'].float().to(device), data['mask'].float().to(device), data['ids']
            pred = model(node_feature, coordinates, mask)

            preds.append(pred)
            node_features.append(node_feature)
            geomouts.append(geomout)
            ids.append(id)

    return preds, geomouts, node_features, ids

def rmsds_on_val_set(preds, geomouts, node_features, decoys=5):
    '''
    Gets rmsds for each fab in the validation set and each decoy
    '''
    CDRs = ["H1", "H2", "H3", "L1", "L2", "L3"]
    cdr_rmsds = torch.zeros(100, decoys, 6)

    for i in range(len(preds)): # loop through the predictions
        for j in range(decoys):
            pred = preds[i][j,:,:,:]
            pred_in = rearrange(pred, "b i d -> () b i d")
            cdr_rmsd = rmsd_per_cdr(pred_in, node_features[i], geomouts[i], CDRs)
            cdr_rmsds[i,j,:] = cdr_rmsd

    return cdr_rmsds

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

# need to get atom_type, amino_type, coords
num2atom = {0: 'CA', 1: 'C', 2: 'N', 3: 'CB'}
num2loop = {0: 'H1', 1: 'H2', 2: 'H3', 3: 'L1', 4: 'L2', 5: 'L3', 6: 'Anchor'}
loop2num = {'H1': 0, 'H2': 1, 'H3': 2, 'L1': 3, 'L2': 4, 'L3': 5, 'Anchor': 6}

def reverse_one_hot(one_hot):
    '''
    Decodes a one hot encoded vector.
    '''
    for i in range(len(one_hot)):
        if one_hot[i] == 1:
            return i

    # happens when it gets to the padded part of the tensor
    return None

def get_atom_type(node_features):
    atom_encoding = node_features[-4:]
    a_num = reverse_one_hot(atom_encoding)
    if a_num == None:
        return None
    return num2atom[a_num]

def get_amino_type(node_features):
    amino_encoding = node_features[:20]
    aa_num = reverse_one_hot(amino_encoding)
    if aa_num == None:
        return None
    return short2long[num2short[aa_num]]

def get_loop(node_features):
    loop_encoding = node_features[30:37]
    l_num = reverse_one_hot(loop_encoding)
    if l_num == None:
        return None
    return num2loop[l_num]

# functions to produce pdb files for model outputs
def produce_pdb_text(coordinates, node_features, chain_name, CDRs):
    '''
    writes a pdb text for one predicted fab
    '''
    # coordinates 504 x 3
    # nodefeatures 504 x 41
    CDRs.append('Anchor')
    text = []
    j = 0
    for i in range(len(node_features)):
        atom_type = get_atom_type(node_features[i])
        amino_type = get_amino_type(node_features[i])
        loop = get_loop(node_features[i])

        if (atom_type == None and amino_type == None):
            continue

        if (atom_type == 'CB' and amino_type == 'GLY'):
            continue

        if loop not in CDRs:
            continue
        
        j += 1
        res_id = math.floor(i / 4)
        line = to_pdb_line(j, atom_type, amino_type, chain_name, res_id, coordinates[i])

        text.append(line)
    
    return text

def pdb_for_set(predictions, geomouts, node_features, ids, out_dir, CDRs=['H1', 'H2', 'H3', 'L1', 'L2', 'L3']):
    '''
    Produces pdb files for the predictions of the validation set. Decoys and ground truth are chains in a single file.
    '''
    n_anchors_before = loop2num[CDRs[0]] * 4 * 4
    n_anchors_after = (5 - loop2num[CDRs[-1]]) * 4 * 4

    if n_anchors_after == 0: # if slice is given index -0 the slice is empty
        n_anchors_after = 1
    
    for i in range(len(predictions)):

        prediction = predictions[i]
        geomout = geomouts[i]
        node_feature = node_features[i]

        

        pdb_text = []
        chains = ['D', 'P', 'Q', 'R', 'S', 'T']
        j = 0
        pdb_text += produce_pdb_text(geomout.squeeze(), node_feature.squeeze(), chains[j], CDRs)[n_anchors_before:-n_anchors_after]
        for decoy in prediction:
            j += 1
            pdb_text += produce_pdb_text(decoy.squeeze(), node_feature.squeeze(), chains[j], CDRs)[n_anchors_before:-n_anchors_after]

        
        pdb_text = ''.join(pdb_text) +'END'

        outfile = 'pdbs/'+out_dir+'/'+ids[i][0][0]+ids[i][1]['HC'][0]+ids[i][1]['LC'][0]+'.pdb'
        with open(outfile, 'w') as f:
            f.write(pdb_text)
