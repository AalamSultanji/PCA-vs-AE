import torch
import torch.nn as nn
import os
from metrics import mse, downstream_acc, r2
import copy

def train_ae(model, train_set, val_split, criterion, optimizer, device, epochs = 50):
    '''
    Training function for the model.
    Arguments: 
    model: the model to be trained
    train_split: Subset for training data
    val_split: Subset for validation data
    criterion: loss function
    optimizer: optimizer for the model (I will use Adam for this exercise)
    device: device to run the model on (CPU or GPU)
    epochs: number of epochs to train the model for
    Returns:
    model: the trained model
    history: a dictionary containing the training and validation loss and accuracy for each epoch
    '''
    #add flattening
    train_losses = []
    val_losses = []
    os.makedirs('state', exist_ok=True)
    early_stopper = EarlyStopping(patience = 3, min_delta = 0.001)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for inputs, _ in train_set:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_set)
        train_losses.append(avg_train_loss)
        #validation part
        val_loss = evaluate(model, val_split, criterion, device)
        if val_loss < min(val_losses):
            best_model = copy.deepcopy(model.state_dict())
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs} || Train Loss: {avg_train_loss:.4f} || Val Loss: {val_loss:.4f}')

        #patience based early stopping
        if early_stopper.early_stop(val_loss):
            print(f'Early stopping at epoch {epoch+1}')
            break
    model.load_state_dict(best_model)   

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    return model, history

def evaluate(model, split, criterion, device):
    '''
    Evaluation for the model
    '''
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, _ in split:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item()
    avg_loss = running_loss / len(split)
    return avg_loss

class EarlyStopping:
    def __init__(self, patience = 5, min_delta = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.min_val_loss = float('inf')
        self.counter = 0
    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    