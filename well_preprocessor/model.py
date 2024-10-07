# name : Hadrian Fung
# Github username : gems-hf923
# model.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv

# Pure U-GCN model
class UNetGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UNetGCN, self).__init__()
        self.encoder1 = GCNConv(input_dim, hidden_dim)  
        self.encoder2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.encoder3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        
        self.decoder1 = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        
        self.final_linear = nn.Linear(hidden_dim * 8, output_dim)
        
        self.skip_conv1 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1)
        self.skip_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        self.act = F.gelu  # Use GELU instead of ReLU

    def forward(self, x, edge_index):
        
        x1 = self.act(self.encoder1(x, edge_index))
        x2 = self.act(self.encoder2(x1, edge_index))
        x3 = self.act(self.encoder3(x2, edge_index))
        
        x3 = x3.unsqueeze(2)
        
        x = self.decoder1(x3)
        x2 = self.skip_conv1(x2.unsqueeze(2)).repeat(1, 1, x.size(2))
        x = x + x2
        
        x = self.decoder2(x)
        x1 = self.skip_conv2(x1.unsqueeze(2)).repeat(1, 1, x.size(2))
        x = x + x1
        
        x = self.decoder3(x)
        
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        
        return x

# Hybrid model with GCN and GAT layers
class UNetGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UNetGAT, self).__init__()
        
        # Encoder layers: First two are GCN, third is GAT with nhead=4
        self.encoder1 = GCNConv(input_dim, hidden_dim)
        
        self.encoder2 = GCNConv(hidden_dim, hidden_dim * 2)
        
        self.encoder3 = GATConv(hidden_dim * 2, hidden_dim * 2, heads=4)  # nhead=4
        # The output dimension here is hidden_dim * 2 * 4
        
        # Decoder layers: First two are GCN, third is GAT with nhead=4
        # Adjust the input dimension to match the output of the encoder3 layer
        self.decoder1 = GCNConv(hidden_dim * 2 * 4, hidden_dim * 2)
        
        self.decoder2 = GCNConv(hidden_dim * 2, hidden_dim)
        
        self.decoder3 = GATConv(hidden_dim, hidden_dim, heads=4)  # nhead=4
        
        # Final dense layers
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 4)  # Adjust the input to match the GAT output
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.final_linear = nn.Linear(hidden_dim * 2, output_dim)
        
        # Activation function
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        
        # Encoder path
        x1 = self.act(self.encoder1(x, edge_index))
        x2 = self.act(self.encoder2(x1, edge_index))
        x3 = self.act(self.encoder3(x2, edge_index))
        
        # Decoder path
        x = self.act(self.decoder1(x3, edge_index))
        x = x + x2  # Skip connection
        
        x = self.act(self.decoder2(x, edge_index))
        x = x + x1  # Skip connection
        
        x = self.act(self.decoder3(x, edge_index))
        
        # Flatten and apply final dense layers
        x = x.view(x.size(0), -1)  # Flattening
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.final_linear(x)
        
        return x

# Multi-hop GCN model
class MultiHopGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, K=2, **kwargs):
        super(MultiHopGCNConv, self).__init__(in_channels, out_channels, **kwargs)
        self.K = K
        self.final_linear = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0), edge_attr=edge_weight, fill_value=1)
        initial_x = x.clone()
        for hop in range(self.K):
            x = super(MultiHopGCNConv, self).forward(initial_x, edge_index, edge_weight)
        x = self.final_linear(x)
        return x

class MulHop_UNetGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, K=2):
        super(UNetGCN, self).__init__()
        self.encoder1 = MultiHopGCNConv(input_dim, hidden_dim, K=K)
        self.encoder2 = MultiHopGCNConv(hidden_dim, hidden_dim * 2, K=K)
        self.encoder3 = MultiHopGCNConv(hidden_dim * 2, hidden_dim * 4, K=K)
        
        self.decoder1 = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        
        self.final_linear = nn.Linear(hidden_dim * 8, output_dim)
        
        self.skip_conv1 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1)
        self.skip_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x1 = self.act(self.encoder1(x, edge_index))
        x2 = self.act(self.encoder2(x1, edge_index))
        x3 = self.act(self.encoder3(x2, edge_index))
        
        x3 = x3.unsqueeze(2)
        
        x = self.decoder1(x3)
        x2 = self.skip_conv1(x2.unsqueeze(2)).repeat(1, 1, x.size(2))
        x = x + x2
        
        x = self.decoder2(x)
        x1 = self.skip_conv2(x1.unsqueeze(2)).repeat(1, 1, x.size(2))
        x = x + x1
        
        x = self.decoder3(x)
        
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        
        return x
