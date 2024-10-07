# name : Hadrian Fung
# Github username : gems-hf923
# MLops_util.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_model(model, train_dataloader, test_dataloader, num_epochs=50, lr=0.001, save_path='best_model.pth', loss_plot_path = 'loss_plot.png', scheduler_patience=3, early_stopping_patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_dataloader:
            batch_loss = 0.0
            for i in range(len(batch)):
                data = batch[i]
                x = data.x[:,[0,1,5,6,7]].to(device)  # Exclude coordinates 
                edge_index = data.edge_index.to(device)
                target = data.y.to(device)
                
                optimizer.zero_grad()
                output = model(x, edge_index)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
            epoch_loss += batch_loss / len(batch)

        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                batch_loss = 0.0
                for i in range(len(batch)):
                    data = batch[i]
                    x = data.x[:,[0,1,5,6,7]].to(device)  
                    edge_index = data.edge_index.to(device)
                    target = data.y.to(device)

                    output = model(x, edge_index)
                    loss = criterion(output, target)
                    batch_loss += loss.item()
                test_loss += batch_loss / len(batch)
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss)
            print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.4f}')

        # Step with scheduler
        scheduler.step(test_loss)

        # Check early stopping
        early_stopping(test_loss, model, save_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(loss_plot_path)

    return model

def calculate_r2(dataloader, dataset, model, device):
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch)):
                data = batch[i]
                x = data.x[:,[0,1,5,6,7]].to(device)
                edge_index = data.edge_index.to(device)
                target = data.y.to(device)
                
                output = model(x, edge_index)
                
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    mean_temp, std_temp = dataset.mean_temp.cpu().numpy(), dataset.std_temp.cpu().numpy()
    mean_pressure, std_pressure = dataset.mean_pressure.cpu().numpy(), dataset.std_pressure.cpu().numpy()

    predictions_temp = predictions[:, 0] * std_temp + mean_temp
    predictions_pressure = predictions[:, 1] * std_pressure + mean_pressure

    targets_temp = targets[:, 0] * std_temp + mean_temp
    targets_pressure = targets[:, 1] * std_pressure + mean_pressure

    r2_temp = r2_score(targets_temp, predictions_temp)
    r2_pressure = r2_score(targets_pressure, predictions_pressure)

    return r2_temp, r2_pressure

def evaluate_model(model, train_dataloader, test_dataloader, train_dataset, test_dataset, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load the model state
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    # Calculate R² for the train set
    train_r2_temperature, train_r2_pressure = calculate_r2(train_dataloader, train_dataset, model, device)
    print(f"Train R² Score - Temperature: {train_r2_temperature:.4f}, Pressure: {train_r2_pressure:.4f}")

    # Calculate R² for the test set
    test_r2_temperature, test_r2_pressure = calculate_r2(test_dataloader, test_dataset, model, device)
    print(f"Test R² Score - Temperature: {test_r2_temperature:.4f}, Pressure: {test_r2_pressure:.4f}")
