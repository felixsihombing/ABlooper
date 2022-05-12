# functions to predict on val set and calculate RMSDs
from retrain_ablooper import *
import torch
import math
from rich.progress import track
from ABlooper.utils import filt
try:
    from ABlooper.openmm_refine import openmm_refine
except ModuleNotFoundError:
    print('Cannot do refinement')

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
    cdr_rmsds = torch.zeros(len(preds), decoys, 6)

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

        if j > 1: # there are several decoys add mean
            pdb_text += produce_pdb_text(prediction.mean(0).squeeze(), node_feature.squeeze(), 'M', CDRs)[n_anchors_before:-n_anchors_after]
        
        pdb_text = ''.join(pdb_text) +'END'

        outfile = 'pdbs/'+out_dir+'/'+ids[i][0][0]+ids[i][1]['HC'][0]+ids[i][1]['LC'][0]+'.pdb'
        with open(outfile, 'w') as f:
            f.write(pdb_text)

def get_info_from_id(id):
    '''
    Return pdb id, chain names and path to the pdb file from an id.
    '''
    pdb_id = id[0][0]
    heavy_c = id[1]['HC'][0]
    light_c = id[1]['LC'][0]
    path = db.fetch(pdb_id).filepath
    pdb_file = "/".join(path.split("/")[:-1] + ["imgt"] + path.split("/")[-1:])

    return pdb_id, heavy_c, light_c, pdb_file

def get_framework_info(pdb_text, chains):
    CDR_with_anchor_slices = {
        "H1": (chains[0], (25, 40)),
        "H2": (chains[0], (54, 67)),
        "H3": (chains[0], (103, 119)),
        "L1": (chains[1], (25, 40)),
        "L2": (chains[1], (54, 67)),
        "L3": (chains[1], (103, 119))}

    atoms = ["CA", "N", "C", "CB"]

    # For all three of these I extract the loop plus two anchors at either side as these are needed for the model.
    CDR_text = {CDR: [x for x in pdb_text if filt(x, *CDR_with_anchor_slices[CDR])] for CDR in
                    CDR_with_anchor_slices}

    CDR_sequences = {
        CDR: "".join([long2short[x.split()[3][-3:]] for x in CDR_text[CDR] if x.split()[2] == "CA"]) for CDR in
        CDR_with_anchor_slices}

    # Here I don't extract the anchors as this is only needed for writing predictions to pdb file.
    CDR_numberings = {CDR: [x.split()[5] for x in CDR_text[CDR] if x.split()[2] == "CA"][2:-2] for CDR in
                           CDR_text}

    CDR_start_atom_id = {CDR: int([x.split()[1] for x in CDR_text[CDR] if x.split()[2] == "N"][2]) for CDR
                              in CDR_text}
    
    return CDR_with_anchor_slices, atoms, CDR_text, CDR_sequences, CDR_numberings, CDR_start_atom_id

def convert_predictions_into_text_for_each_CDR(CDR_start_atom_id, predicted_CDRs, CDR_sequences, CDR_numberings, CDR_with_anchor_slices):
    pdb_format = {}
    pdb_atoms = ["N", "CA", "C", "CB"]

    permutation_to_reorder_atoms = [1, 0, 2, 3]

    for CDR in CDR_start_atom_id:
        new_text = []
        BB_coords = predicted_CDRs[CDR]
        seq = CDR_sequences[CDR][2:-2]
        numbering = CDR_numberings[CDR]
        atom_id = CDR_start_atom_id[CDR]
        chain = CDR_with_anchor_slices[CDR][0]
        for i, amino in enumerate(BB_coords):
            amino_type = short2long[seq[i]]
            for j, coord in enumerate(amino[permutation_to_reorder_atoms]):
                if (pdb_atoms[j] == "CB") and (amino_type == "GLY"):
                    continue
                new_text.append(to_pdb_line(atom_id, pdb_atoms[j], amino_type, chain, numbering[i], coord))
                atom_id += 1
        pdb_format[CDR] = new_text

    return pdb_format

def pdb_select_hc_lc(pdb_text, chains):
    '''
    Returns only lines of a pdb file which correspond to the heavy or light chain of the antibody
    '''
    atoms = [line for line in pdb_text if line.split()[0] == 'ATOM']
    pdb_text_hc_lc = [line for line in atoms if line.split()[4] in chains]
    
    return pdb_text_hc_lc

def extract_BB_coords(CDR_text, CDR_with_anchor_slices, CDR_sequences, atoms):
    CDR_BB_coords = {}

    for CDR in CDR_with_anchor_slices:
        loop = CDR_text[CDR]

        coors = np.zeros((len(CDR_sequences[CDR]), 4, 3))
        coors[...] = float("Nan")

        i = 0
        res = loop[i].split()[5]

        for line in loop:
            cut = line.split()
            if cut[5] != res:
                res = cut[5]
                i += 1
            if cut[2] in atoms:
                j = atoms.index(cut[2])
                # Using split for coords doesn't always work. Following Biopython approach:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coors[i, j] = np.array([x, y, z])

        # If missed CB (GLY) then add CA instead
        coors[:, 3] = np.where(np.all(coors[:, 3] != coors[:, 3], axis=-1, keepdims=True), coors[:, 0], coors[:, 3])
        CDR_BB_coords[CDR] = coors

    return CDR_BB_coords
    
def produce_full_structures_of_val_set(val_dataloader, model, outdir='', relax=True, to_be_rewritten=["H1", "H2", "H3", "L1", "L2", "L3"]):
    '''
    Produces full FAB structure for a dataset
    '''
    CDR_rmsds_not_relaxed = list()
    CDR_rmsds_relaxed = list()
    decoy_diversities = list()
    order_of_pdbs = list()

    with torch.no_grad():
        model.eval()

        for data in track(val_dataloader, description='predict val set'):

            # predict sturcture using the model
            coordinates, geomout, node_feature, mask, id = data['geomins'].float().to(device), data['geomouts'].float().to(device), data['encodings'].float().to(device), data['mask'].float().to(device), data['ids']
            pred = model(node_feature, coordinates, mask)
            CDR_rmsds_not_relaxed.append(rmsd_per_cdr(pred, node_feature, geomout).tolist())
            pred = pred.squeeze() # remove batch dimension
            
            # get framework info from pdb file
            pdb_id, heavy_c, light_c, pdb_file = get_info_from_id(id)
            chains = [heavy_c, light_c]
            order_of_pdbs.append(pdb_id)

            with open(pdb_file) as file:
                pdb_text = [line for line in file.readlines()]
                
            pdb_text = pdb_select_hc_lc(pdb_text, chains)

            CDR_with_anchor_slices, atoms, CDR_text, CDR_sequences, CDR_numberings, CDR_start_atom_id = get_framework_info(pdb_text, chains)

            predicted_CDRs = {}
            all_decoys = {}
            decoy_diversity = {}

            for i, CDR in enumerate(CDR_with_anchor_slices):
                output_CDR = pred[:, node_feature[0, :, 30 + i] == 1.0]
                all_decoys[CDR] = rearrange(output_CDR, "b (i a) d -> b i a d", a=4).cpu().numpy()
                predicted_CDRs[CDR] = rearrange(output_CDR.mean(0), "(i a) d -> i a d", a=4).cpu().numpy()
                decoy_diversity[CDR] = (output_CDR[None] - output_CDR[:, None]).pow(2).sum(-1).mean(-1).pow(
                    1 / 2).sum().item() / 20
            
            decoy_diversities.append(list(decoy_diversity.values()))
            
            text_prediction_per_CDR = convert_predictions_into_text_for_each_CDR(CDR_start_atom_id, predicted_CDRs, CDR_sequences, CDR_numberings, CDR_with_anchor_slices)
            old_text = pdb_text

            for CDR in to_be_rewritten:
                new = True
                new_text = []
                chain, CDR_slice = CDR_with_anchor_slices[CDR]
                CDR_slice = (CDR_slice[0] + 2, CDR_slice[1] - 2)

                for line in old_text:
                    if not filt(line, chain, CDR_slice):
                        new_text.append(line)
                    elif new:
                        new_text += text_prediction_per_CDR[CDR]
                        new = False
                    else:
                        continue
                old_text = new_text

            header = [
                "REMARK    CDR LOOPS REMODELLED USING ABLOOPER                                   \n"]
            new_text = header + old_text

            with open('pdbs/'+outdir+'/'+pdb_id+'-'+heavy_c+light_c+'.pdb', "w+") as file:
                file.write("".join(new_text))

            with open('pdbs/'+outdir+'/'+pdb_id+'-'+heavy_c+light_c+'-true.pdb', "w+") as file:
                file.write("".join(pdb_text))

            if relax:
                relaxed_text = openmm_refine(old_text, CDR_with_anchor_slices)
                header.append("REMARK    REFINEMENT DONE USING OPENMM" + 42 * " " + "\n")
                relaxed_text = header + relaxed_text

                with open('pdbs/'+outdir+'/'+pdb_id+'-'+heavy_c+light_c+'-relaxed.pdb', "w+") as file:
                    file.write(''.join(relaxed_text))


                # calculate rmsds of relaxed structures
                CDR_with_anchor_slices, atoms, CDR_text, CDR_sequences, CDR_numberings, CDR_start_atom_id = get_framework_info(relaxed_text, chains)
                CDR_BB_coords = extract_BB_coords(CDR_text, CDR_with_anchor_slices, CDR_sequences, atoms)

                relaxed_coords = prepare_model_output([CDR_BB_coords])[0]
            
                relaxed_coords = pad_tensor(relaxed_coords)
                relaxed_coords = rearrange(relaxed_coords, 'i x -> () () i x')
                CDR_rmsds_relaxed.append(rmsd_per_cdr(relaxed_coords, node_feature, geomout).tolist())


    return CDR_rmsds_not_relaxed, CDR_rmsds_relaxed, decoy_diversities, order_of_pdbs
