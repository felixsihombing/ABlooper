'''Functions to train the model. To versions provided to train the model with mask and the one without'''

from einops import rearrange
import torch
import numpy as np
import json
from rich.progress import track
from retrain_ablooper import *

# torch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)

# train for one epoch
def run_epoch(model, optim, train_dataloader, val_dataloader, grad_clip=10.0):
    '''
    Function to train model for a single epoch
    '''
    epoch_train_losses = []
    epoch_val_losses = []
    model.train()                                                      # Set the model to train mode (Should't matter here as we don't have dropout, but good practice to keep in)

    for i,data in enumerate(train_dataloader):                         # For each batch of data in the dataset
        coordinates, geomouts, node_features = data['geomins'].float(), data['geomouts'].float(), data['encodings'].float()

        pred = model(node_features, coordinates)
        optim.zero_grad()                                              # Delete old gradients

        loss = rmsd(geomouts, pred)
        epoch_train_losses.append(loss.item())                         # Store value of loss function for training set

        loss.backward()                                                # Calculate loss gradients (pytorch handles this in the background)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # Optional: Clip the norm of the gradient (It stops the optimiser from doing very large updates at once)
        optim.step()                                                   # Update model weights

    with torch.no_grad():                                              # Calculate loss funtion for validation set
        model.eval()                                                   # Set the model to eval mode
        for i, data in enumerate(val_dataloader):
            coordinates, geomouts, node_features, mask = data['geomins'].float(), data['geomouts'].float(), data['encodings'].float(), data['mask'].float()
            pred = model(node_features, coordinates, mask)
            loss = rmsd(geomouts, pred)
            epoch_val_losses.append(loss.item())
    
    return np.mean(epoch_train_losses), np.mean(epoch_val_losses)

# train for one epoch with mask
def mask_run_epoch(model, optim, train_dataloader, val_dataloader, decoys=5, grad_clip=10.0, ):
    '''
    Function to train model for a single epoch with mask
    '''
    CDRs = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']
    cdr_rmsds = torch.zeros(100, len(CDRs))

    epoch_train_losses = []
    epoch_val_losses = []
    model.train()                                                      # Set the model to train mode (Should't matter here as we don't have dropout, but good practice to keep in)

    for i,data in enumerate(train_dataloader):                         # For each batch of data in the dataset
        coordinates, geomouts, node_features, mask = data['geomins'].float().to(device), data['geomouts'].float().to(device), data['encodings'].float().to(device), data['mask'].float().to(device)

        pred = model(node_features, coordinates, mask)
        optim.zero_grad()                                              # Delete old gradients

        loss = rmsd(geomouts, pred)
        epoch_train_losses.append(loss.item())                         # Store value of loss function for training set

        loss.backward()                                                # Calculate loss gradients (pytorch handles this in the background)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # Optional: Clip the norm of the gradient (It stops the optimiser from doing very large updates at once)
        optim.step()                                                   # Update model weights

    with torch.no_grad():                                              # Calculate loss funtion for validation set
        model.eval()                                                   # Set the model to eval mode
        for i, data in enumerate(val_dataloader):
            coordinates, geomouts, node_features, mask = data['geomins'].float().to(device), data['geomouts'].float().to(device), data['encodings'].float().to(device), data['mask'].float().to(device)
            pred = model(node_features, coordinates, mask)
            loss = rmsd(geomouts, pred)
            epoch_val_losses.append(loss.item())

            cdr_rmsds[i,:] = rmsd_per_cdr(pred, node_features, geomouts, CDRs, decoys)
    
    return np.mean(epoch_train_losses), np.mean(epoch_val_losses), cdr_rmsds.mean(0)

def train_model(model, optimiser, train_dataloader, val_dataloader, training_name='', n_epochs=5000, patience=150, decoys=5):
    '''
    Runs run_epoch function a specified number of times and keeps track of loss.
    '''
    train_losses = []
    val_losses = []
    cdr_rmsds = []

    print(" Train |  Val ")
    for epoch in track(range(n_epochs), description='Train model'):
        train_loss, val_loss, cdr_rmsd = mask_run_epoch(model, optimiser, train_dataloader, val_dataloader, decoys=decoys)  # Run one epoch and get train and validation loss

        train_losses.append(train_loss.tolist())                                                    # Store train and validation loss
        val_losses.append(val_loss.tolist())
        cdr_rmsds.append(cdr_rmsd.tolist())

        if np.min(val_losses) == val_loss:                                                 # If it is the best model on the validation set, save it
            best_model_name = "best_model" + training_name
            torch.save(model.state_dict(), best_model_name)                                   # This is how you save models in pytorch
            epochs_without_improvement = 0

        elif epochs_without_improvement < patience:                                        # If the model hasn't improved this epoch store that
            epochs_without_improvement += 1
        else:                                                                              # If the model hasn't improved in 'patience' epochs stop the training.
            break
        
        previous_weigths_name = "previous_wieghts" + training_name
        previous_optim_name = "previous_optim" + training_name
        if train_loss > 1.5*np.min(train_losses):                                          # EGNNs are quite unstable, this reverts the model to a previous state if an epoch blows up
            model.load_state_dict(torch.load(previous_weigths_name, map_location=torch.device(device)))
            optimiser.load_state_dict(torch.load(previous_optim_name, map_location=torch.device(device)))
        if train_loss == np.min(train_losses):
            torch.save(model.state_dict(), previous_weigths_name)        
            torch.save(optimiser.state_dict(), previous_optim_name)

        losses_file = "training_loss" + training_name + ".json" 
        with open(losses_file, 'w') as f:
            dic = {'train_losses': train_losses,
                   'val_lossers': val_losses,
                   'cdr_rmsd': cdr_rmsds}
            json.dump(dic, f)


        print("{:6.2f} | {:6.2f}".format(train_loss, train_loss))
    
    return train_losses, val_losses
