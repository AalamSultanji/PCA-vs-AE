import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, f1_score, r2_score
import copy

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs = 50):
    '''
    Training function for the model.
    Arguments: 
    model: the model to be trained
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
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
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += inputs.size(0)
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        #validation part
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        if val_loss < min(val_losses):
            best_model = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch+1}/{epochs} || Train Loss: {avg_train_loss:.4f} || Val Loss: {val_loss:.4f}')

        #patience based early stopping
        if early_stopper.early_stop(val_loss):
            print(f'Early stopping at epoch {epoch+1}')
            break
    model = model.load_state_dict(best_model)   

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    return model, history

def evaluate(model, loader, criterion, device):
    '''
    Evaluation for the model
    '''
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    return avg_loss

def test(model, test_loader, device):
    '''
    Testing function for the model, computes the confusion matrix and classification report. 
    '''
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            model.encode(inputs)
    
    acc_score = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    r2 = r2_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc_score:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Test R2 Score: {r2:.4f}')
    

    return {
        'accuracy': acc_score,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'r2_score': r2
    }

class EarlyStopping:
    def __init__(self, patience = 5, min_delta = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.min_val_loss = float('inf')
        self.counter = 0
    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delt):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    