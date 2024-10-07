# name : Hadrian Fung
# Github username : gems-hf923
# vtu_visualization_util.py

import meshio
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from .dataset_preprocessing import *
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from .dataset_preprocessing import *

def convert_npy_to_vtu(npy_file_path, vtu_file_path):
    all_points = np.load(npy_file_path)

    points = all_points[:, :3]
    phase1_pressure = all_points[:, 3]
    phase1_temperature = all_points[:, 4]
    phase2_pressure = all_points[:, 5]
    phase2_temperature = all_points[:, 6]

    point_data = {
        'phase1::Pressure': phase1_pressure,
        'phase1::Temperature': phase1_temperature,
        'phase2::Pressure': phase2_pressure,
        'phase2::Temperature': phase2_temperature
        # 'well_points': well_points,
        # 'screen_points': screen_points
    }

    num_points = len(points)
    cells = []
    for i in range(0, num_points - 3, 4):
        cells.append([i, i+1, i+2, i+3])

    cell_type = [("tetra", np.array(cells))]
    mesh = meshio.Mesh(points, cell_type, point_data=point_data)
    mesh.write(vtu_file_path)
    print(f"Converted {npy_file_path} to {vtu_file_path}")

def plot_3d_scatter(data, title):
    x, y, z, temp, pres = data[:, 2], data[:, 3], data[:, 4], data[:, 0], data[:, 1]  # Adjust indices for correct temperature mapping

    scatter = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=3, color=temp, colorscale='RdYlBu_r', opacity=0.8, colorbar=dict(title='Temperature'))
    )
    layout = go.Layout(title=title, scene=dict(
        xaxis=dict(title='X', range=[-250, 250]),  # Adjust range based on your data
        yaxis=dict(title='Y', range=[-250, 250]),  # Adjust range based on your data
        zaxis=dict(title='Z', range=[-250, 0])   # Adjust range based on your data
    ))
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

    # clear figure
    plt.clf()

    scatter = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=3, color=pres, colorscale='RdYlBu_r', opacity=0.8, colorbar=dict(title='Temperature'))
    )
    layout = go.Layout(title=title, scene=dict(
        xaxis=dict(title='X', range=[-250, 250]),  # Adjust range based on your data
        yaxis=dict(title='Y', range=[-250, 250]),  # Adjust range based on your data
        zaxis=dict(title='Z', range=[-250, 0])   # Adjust range based on your data
    ))
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

def fwd_plot_cross_section(graphs, model, mean, std, device='cpu', y_value=0, epsilon=10, dims=200, mode='temp'):
    n_graphs = len(graphs)
    
    # Exit early if there are less than 2 graphs
    if n_graphs < 2:
        print("Not enough graphs to plot. Exiting function.")
        return

    if mode == 'temp':
        pred_idx = 0
        vmin = 8
        vmax = 18
    elif mode == 'pres':
        pred_idx = 1
        vmin = 2.5e6
        vmax = -2.5e6

    model.eval()
    fig, axes = plt.subplots(n_graphs - 1, 2, figsize=(14, 9 * (n_graphs - 1)), dpi=300)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D

    for i in range(n_graphs - 1):  # Loop only for n-1 graphs
        graph = graphs[i].to(device)
        with torch.no_grad():
            x = graph.x[:,[0,1,5,6,7]].to(device)
            edge_index = graph.edge_index.to(device)
            target = graph.y[:, pred_idx].to(device) if graph.y is not None else None
            
            output = model(x, edge_index)[:, pred_idx]
            predictions = (output.cpu().numpy().ravel() * std + mean)
            targets = (target.cpu().numpy().ravel() if target is not None else None)
            
            coords = graph.x[:, 2:5].cpu().numpy()
            mask = np.abs(coords[:, 1] - y_value) < epsilon
            x_coords = coords[mask, 0]
            z_coords = coords[mask, 2]
            predicted_temp = predictions[mask]
            true_temp = targets[mask] if targets is not None else None
            # Plot predicted temperature
            scatter_pred = axes[i, 0].scatter(x_coords, z_coords, c=predicted_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
            fig.colorbar(scatter_pred, ax=axes[i, 0])
            axes[i, 0].set_title(f"Predicted {mode} at Timestep {i+1}")
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Z')
            axes[i, 0].set_xlim([-dims, dims])
            axes[i, 0].set_ylim([-250, 0])

            # Plot true temperature on the current graph
            if true_temp is not None:
                scatter_true = axes[i, 1].scatter(x_coords, z_coords, c=true_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
                fig.colorbar(scatter_true, ax=axes[i, 1])
                axes[i, 1].set_title(f"True {mode} at Timestep {i+1} | Projected mesh")
                axes[i, 1].set_xlabel('X')
                axes[i, 1].set_ylabel('Z')
                axes[i, 1].set_xlim([-dims, dims])
                axes[i, 1].set_ylim([-250, 0])

    plt.tight_layout()
    plt.show()

def back_plot_cross_section(graphs, model, mean, std, device='cpu', y_value=0, epsilon=10, dims=200, mode='temp'):
    n_graphs = len(graphs)
    
    # Exit early if there are less than 2 graphs
    if n_graphs < 2:
        print("Not enough graphs to plot. Exiting function.")
        return

    if mode == 'temp':
        pred_idx = 0
        vmin = 8
        vmax = 18
    elif mode == 'pres':
        pred_idx = 1
        vmin = 2.5e6
        vmax = -2.5e6

    model.eval()
    fig, axes = plt.subplots(n_graphs - 1, 3, figsize=(18, 9 * (n_graphs - 1)), dpi=300)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D
    
    for i in range(n_graphs - 1):  # Loop only for n-1 graphs
        graph = graphs[i].to(device)
        with torch.no_grad():
            x = graph.x[:,[0,1,5,6,7]].to(device)
            edge_index = graph.edge_index.to(device)
            target = graph.y[:, pred_idx].to(device) if graph.y is not None else None
            
            output = model(x, edge_index)[:, pred_idx]
            predictions = (output.cpu().numpy().ravel() * std + mean)
            targets = (target.cpu().numpy().ravel() if target is not None else None)
            
            coords = graph.x[:, 2:5].cpu().numpy()
            mask = np.abs(coords[:, 1] - y_value) < epsilon
            x_coords = coords[mask, 0]
            z_coords = coords[mask, 2]
            predicted_temp = predictions[mask]
            true_temp = targets[mask] if targets is not None else None
            # Plot predicted temperature
            scatter_pred = axes[i, 0].scatter(x_coords, z_coords, c=predicted_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
            fig.colorbar(scatter_pred, ax=axes[i, 0])
            axes[i, 0].set_title(f"Predicted {mode} at Timestep {i+1}")
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Z')
            axes[i, 0].set_xlim([-dims, dims])
            axes[i, 0].set_ylim([-250, 0])

            # Plot true temperature on the current graph
            if true_temp is not None:
                scatter_true = axes[i, 1].scatter(x_coords, z_coords, c=true_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
                fig.colorbar(scatter_true, ax=axes[i, 1])
                axes[i, 1].set_title(f"True {mode} at Timestep {i+1} | Projected mesh")
                axes[i, 1].set_xlabel('X')
                axes[i, 1].set_ylabel('Z')
                axes[i, 1].set_xlim([-dims, dims])
                axes[i, 1].set_ylim([-250, 0])

            # Plot true temperature on the next graph's mesh
            next_graph = graphs[i + 1]
            next_coords = next_graph.x[:, 2:5].cpu().numpy()
            next_temp = (next_graph.x[:, pred_idx].cpu().numpy() * std + mean)
            mask_next = np.abs(next_coords[:, 1] - y_value) < epsilon
            next_x_coords = next_coords[mask_next, 0]
            next_z_coords = next_coords[mask_next, 2]
            next_temp = next_temp[mask_next]
            
            scatter_next_true = axes[i, 2].scatter(next_x_coords, next_z_coords, c=next_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
            fig.colorbar(scatter_next_true, ax=axes[i, 2])
            axes[i, 2].set_title(f"True {mode} at Timestep {i+1} | Original mesh")
            axes[i, 2].set_xlabel('X')
            axes[i, 2].set_ylabel('Z')
            axes[i, 2].set_xlim([-dims, dims])
            axes[i, 2].set_ylim([-250, 0])

    plt.tight_layout()
    plt.show()

class SingleScenarioDataset(InMemoryDataset):
    def __init__(self, root, scenario, max_time=5, projection_method='backward'):
        self.scenario = f'scenario{scenario}'
        self.scenario_id = scenario
        self.max_time = max_time  
        self.projection_method = projection_method
        
        # Set this to None initially, and it will be set after processing/loading
        self._data_list = None
        
        # Call parent class constructor with only necessary parameters
        super(SingleScenarioDataset, self).__init__(root)
        
        # After parent constructor completes, check if data is loaded or needs normalization
        if self._data_list is None:
            # Load processed data
            self._data_list = torch.load(self.processed_paths[0])
            # Normalize the data
            self.normalize_data()

    @property
    def raw_file_names(self):
        all_files = []
        scenario_path = os.path.join(self.raw_dir, f'scenario{self.scenario}')
        if os.path.isdir(scenario_path) and not scenario_path.startswith('.'):
            for timestep in range(1, self.max_time + 1):
                filename = f"z_ATES_{self.scenario}_{timestep}.vtu"
                all_files.append(filename)
        return all_files

    @property
    def processed_file_names(self):
        return [f'data{self.scenario}_Head{self.max_time}.pt']

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def process(self):
        scenario_path = os.path.join(self.raw_dir, f'{self.scenario}')
        injection_sequence = generate_injection_sequence()

        scenario_graphs = []
        for timestep in range(1, self.max_time + 1):
            file_path = os.path.join(scenario_path, f"z_ATES_{self.scenario}_{timestep}.vtu")
            print(f"Looking for file: {file_path}")
            if os.path.exists(file_path):
                print(f"Processing scenario {self.scenario} at timestep {timestep}...")
                vtu_0 = read_vtu(file_path)
                node_data = vtu_0.point_data
                cell_data = vtu_0.cell_data_dict
                cells = np.vstack([cell.data for cell in vtu_0.cells if cell.type == "tetra"])
                edge_data = project_cell_data_to_edges(cells, vtu_0.points)
                graph = create_graph(vtu_0.points, node_data, edge_data, self.scenario_id, timestep, injection_sequence)
                scenario_graphs.append(graph)
            else:
                print(f"File not found: {file_path}")

        for t in range(len(scenario_graphs) - 1):
            graph_t = scenario_graphs[t]
            graph_t_plus_1 = scenario_graphs[t + 1]
            if self.projection_method == 'backward':
                projected_graph = project_graph_to_previous_mesh(graph_t, graph_t_plus_1)
            elif self.projection_method == 'forward':
                projected_graph = project_graph_to_next_mesh(graph_t, graph_t_plus_1)
            scenario_graphs[t] = projected_graph

        if len(scenario_graphs) == 0:
            raise ValueError("Insufficient data to process. Ensure the scenario has at least one timesteps.")

        torch.save(scenario_graphs, self.processed_paths[0])

        self._data_list = scenario_graphs

    def normalize_data(self):
        # Compute mean and std for temperature and pressure from the data
        raw_temps = [data.x[:, 0].clone() for data in self._data_list if data is not None]
        raw_temps = torch.cat(raw_temps, dim=0)
        self.mean_temp = raw_temps.mean()
        self.std_temp = raw_temps.std()
        
        raw_pressures = [data.x[:, 1].clone() for data in self._data_list if data is not None]
        raw_pressures = torch.cat(raw_pressures, dim=0)
        self.mean_pressure = raw_pressures.mean()
        self.std_pressure = raw_pressures.std()
        
        # Apply normalization using the computed stats
        self.apply_normalization()

    def apply_normalization(self):
        # Normalize the data using self.mean_temp, self.std_temp, self.mean_pressure, self.std_pressure
        for data in self._data_list:
            if data is not None:
                data.x[:, 0] = (data.x[:, 0] - self.mean_temp) / self.std_temp  # Normalize temperature
                data.x[:, 1] = (data.x[:, 1] - self.mean_pressure) / self.std_pressure  # Normalize pressure

    def unnormalize(self, data):
        data.x[:, 0] = data.x[:, 0] * self.std_temp + self.mean_temp  # Unnormalize temperature
        data.x[:, 1] = data.x[:, 1] * self.std_pressure + self.mean_pressure  # Unnormalize pressure
        return data

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

def plot_3d_scatter(data, title):
    x, y, z, temp, pres = data[:, 2], data[:, 3], data[:, 4], data[:, 0], data[:, 1]  # Adjust indices for correct temperature mapping

    scatter = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=3, color=temp, colorscale='RdYlBu_r', opacity=0.8, colorbar=dict(title='Temperature'))
    )
    layout = go.Layout(title=title, scene=dict(
        xaxis=dict(title='X', range=[-250, 250]),  # Adjust range based on your data
        yaxis=dict(title='Y', range=[-250, 250]),  # Adjust range based on your data
        zaxis=dict(title='Z', range=[-250, 0])   # Adjust range based on your data
    ))
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

    # clear figure
    plt.clf()

    scatter = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=3, color=pres, colorscale='RdYlBu_r', opacity=0.8, colorbar=dict(title='Temperature'))
    )
    layout = go.Layout(title=title, scene=dict(
        xaxis=dict(title='X', range=[-250, 250]),  # Adjust range based on your data
        yaxis=dict(title='Y', range=[-250, 250]),  # Adjust range based on your data
        zaxis=dict(title='Z', range=[-250, 0])   # Adjust range based on your data
    ))
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

def fwd_plot_cross_section(graphs, model, mean, std, device='cpu', y_value=0, epsilon=10, dims=200, mode='temp'):
    n_graphs = len(graphs)
    
    # Exit early if there are less than 2 graphs
    if n_graphs < 2:
        print("Not enough graphs to plot. Exiting function.")
        return

    if mode == 'temp':
        pred_idx = 0
        vmin = 8
        vmax = 18
    elif mode == 'pres':
        pred_idx = 1
        vmin = 2.5e6
        vmax = -2.5e6

    model.eval()
    fig, axes = plt.subplots(n_graphs - 1, 2, figsize=(14, 9 * (n_graphs - 1)), dpi=300)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D

    for i in range(n_graphs - 1):  # Loop only for n-1 graphs
        graph = graphs[i].to(device)
        with torch.no_grad():
            x = graph.x[:,[0,1,5,6,7]].to(device)
            edge_index = graph.edge_index.to(device)
            target = graph.y[:, pred_idx].to(device) if graph.y is not None else None
            
            output = model(x, edge_index)[:, pred_idx]
            predictions = (output.cpu().numpy().ravel() * std + mean)
            targets = (target.cpu().numpy().ravel() if target is not None else None)
            
            coords = graph.x[:, 2:5].cpu().numpy()
            mask = np.abs(coords[:, 1] - y_value) < epsilon
            x_coords = coords[mask, 0]
            z_coords = coords[mask, 2]
            predicted_temp = predictions[mask]
            true_temp = targets[mask] if targets is not None else None
            # Plot predicted temperature
            scatter_pred = axes[i, 0].scatter(x_coords, z_coords, c=predicted_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
            fig.colorbar(scatter_pred, ax=axes[i, 0])
            axes[i, 0].set_title(f"Predicted {mode} at Timestep {i+1}")
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Z')
            axes[i, 0].set_xlim([-dims, dims])
            axes[i, 0].set_ylim([-250, 0])

            # Plot true temperature on the current graph
            if true_temp is not None:
                scatter_true = axes[i, 1].scatter(x_coords, z_coords, c=true_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
                fig.colorbar(scatter_true, ax=axes[i, 1])
                axes[i, 1].set_title(f"True {mode} at Timestep {i+1} | Projected mesh")
                axes[i, 1].set_xlabel('X')
                axes[i, 1].set_ylabel('Z')
                axes[i, 1].set_xlim([-dims, dims])
                axes[i, 1].set_ylim([-250, 0])

    plt.tight_layout()
    plt.show()

def back_plot_cross_section(graphs, model, mean, std, device='cpu', y_value=0, epsilon=10, dims=200, mode='temp'):
    n_graphs = len(graphs)
    
    # Exit early if there are less than 2 graphs
    if n_graphs < 2:
        print("Not enough graphs to plot. Exiting function.")
        return

    if mode == 'temp':
        pred_idx = 0
        vmin = 8
        vmax = 18
    elif mode == 'pres':
        pred_idx = 1
        vmin = 2.5e6
        vmax = -2.5e6

    model.eval()
    fig, axes = plt.subplots(n_graphs - 1, 3, figsize=(18, 9 * (n_graphs - 1)), dpi=300)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D
    
    for i in range(n_graphs - 1):  # Loop only for n-1 graphs
        graph = graphs[i].to(device)
        with torch.no_grad():
            x = graph.x[:,[0,1,5,6,7]].to(device)
            edge_index = graph.edge_index.to(device)
            target = graph.y[:, pred_idx].to(device) if graph.y is not None else None
            
            output = model(x, edge_index)[:, pred_idx]
            predictions = (output.cpu().numpy().ravel() * std + mean)
            targets = (target.cpu().numpy().ravel() if target is not None else None)
            
            coords = graph.x[:, 2:5].cpu().numpy()
            mask = np.abs(coords[:, 1] - y_value) < epsilon
            x_coords = coords[mask, 0]
            z_coords = coords[mask, 2]
            predicted_temp = predictions[mask]
            true_temp = targets[mask] if targets is not None else None
            # Plot predicted temperature
            scatter_pred = axes[i, 0].scatter(x_coords, z_coords, c=predicted_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
            fig.colorbar(scatter_pred, ax=axes[i, 0])
            axes[i, 0].set_title(f"Predicted {mode} at Timestep {i+1}")
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Z')
            axes[i, 0].set_xlim([-dims, dims])
            axes[i, 0].set_ylim([-250, 0])

            # Plot true temperature on the current graph
            if true_temp is not None:
                scatter_true = axes[i, 1].scatter(x_coords, z_coords, c=true_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
                fig.colorbar(scatter_true, ax=axes[i, 1])
                axes[i, 1].set_title(f"True {mode} at Timestep {i+1} | Projected mesh")
                axes[i, 1].set_xlabel('X')
                axes[i, 1].set_ylabel('Z')
                axes[i, 1].set_xlim([-dims, dims])
                axes[i, 1].set_ylim([-250, 0])

            # Plot true temperature on the next graph's mesh
            next_graph = graphs[i + 1]
            next_coords = next_graph.x[:, 2:5].cpu().numpy()
            next_temp = (next_graph.x[:, pred_idx].cpu().numpy() * std + mean)
            mask_next = np.abs(next_coords[:, 1] - y_value) < epsilon
            next_x_coords = next_coords[mask_next, 0]
            next_z_coords = next_coords[mask_next, 2]
            next_temp = next_temp[mask_next]
            
            scatter_next_true = axes[i, 2].scatter(next_x_coords, next_z_coords, c=next_temp, cmap='coolwarm', s=10, vmin=vmin, vmax=vmax)
            fig.colorbar(scatter_next_true, ax=axes[i, 2])
            axes[i, 2].set_title(f"True {mode} at Timestep {i+1} | Original mesh")
            axes[i, 2].set_xlabel('X')
            axes[i, 2].set_ylabel('Z')
            axes[i, 2].set_xlim([-dims, dims])
            axes[i, 2].set_ylim([-250, 0])

    plt.tight_layout()
    plt.show()

class SingleScenarioDataset(InMemoryDataset):
    def __init__(self, root, scenario, max_time=5, projection_method='backward'):
        self.scenario = f'scenario{scenario}'
        self.scenario_id = scenario
        self.max_time = max_time  
        self.projection_method = projection_method
        
        # Set this to None initially, and it will be set after processing/loading
        self._data_list = None
        
        # Call parent class constructor with only necessary parameters
        super(SingleScenarioDataset, self).__init__(root)
        
        # After parent constructor completes, check if data is loaded or needs normalization
        if self._data_list is None:
            # Load processed data
            self._data_list = torch.load(self.processed_paths[0])
            # Normalize the data
            self.normalize_data()

    @property
    def raw_file_names(self):
        all_files = []
        scenario_path = os.path.join(self.raw_dir, f'scenario{self.scenario}')
        if os.path.isdir(scenario_path) and not scenario_path.startswith('.'):
            for timestep in range(1, self.max_time + 1):
                filename = f"z_ATES_{self.scenario}_{timestep}.vtu"
                all_files.append(filename)
        return all_files

    @property
    def processed_file_names(self):
        return [f'data{self.scenario}_Head{self.max_time}.pt']

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def process(self):
        scenario_path = os.path.join(self.raw_dir, f'{self.scenario}')
        injection_sequence = generate_injection_sequence()

        scenario_graphs = []
        for timestep in range(1, self.max_time + 1):
            file_path = os.path.join(scenario_path, f"z_ATES_{self.scenario}_{timestep}.vtu")
            print(f"Looking for file: {file_path}")
            if os.path.exists(file_path):
                print(f"Processing scenario {self.scenario} at timestep {timestep}...")
                vtu_0 = read_vtu(file_path)
                node_data = vtu_0.point_data
                cell_data = vtu_0.cell_data_dict
                cells = np.vstack([cell.data for cell in vtu_0.cells if cell.type == "tetra"])
                edge_data = project_cell_data_to_edges(cells, vtu_0.points)
                graph = create_graph(vtu_0.points, node_data, edge_data, self.scenario_id, timestep, injection_sequence)
                scenario_graphs.append(graph)
            else:
                print(f"File not found: {file_path}")

        for t in range(len(scenario_graphs) - 1):
            graph_t = scenario_graphs[t]
            graph_t_plus_1 = scenario_graphs[t + 1]
            if self.projection_method == 'backward':
                projected_graph = project_graph_to_previous_mesh(graph_t, graph_t_plus_1)
            elif self.projection_method == 'forward':
                projected_graph = project_graph_to_next_mesh(graph_t, graph_t_plus_1)
            scenario_graphs[t] = projected_graph

        if len(scenario_graphs) == 0:
            raise ValueError("Insufficient data to process. Ensure the scenario has at least one timesteps.")

        torch.save(scenario_graphs, self.processed_paths[0])

        self._data_list = scenario_graphs

    def normalize_data(self):
        # Compute mean and std for temperature and pressure from the data
        raw_temps = [data.x[:, 0].clone() for data in self._data_list if data is not None]
        raw_temps = torch.cat(raw_temps, dim=0)
        self.mean_temp = raw_temps.mean()
        self.std_temp = raw_temps.std()
        
        raw_pressures = [data.x[:, 1].clone() for data in self._data_list if data is not None]
        raw_pressures = torch.cat(raw_pressures, dim=0)
        self.mean_pressure = raw_pressures.mean()
        self.std_pressure = raw_pressures.std()
        
        # Apply normalization using the computed stats
        self.apply_normalization()

    def apply_normalization(self):
        # Normalize the data using self.mean_temp, self.std_temp, self.mean_pressure, self.std_pressure
        for data in self._data_list:
            if data is not None:
                data.x[:, 0] = (data.x[:, 0] - self.mean_temp) / self.std_temp  # Normalize temperature
                data.x[:, 1] = (data.x[:, 1] - self.mean_pressure) / self.std_pressure  # Normalize pressure

    def unnormalize(self, data):
        data.x[:, 0] = data.x[:, 0] * self.std_temp + self.mean_temp  # Unnormalize temperature
        data.x[:, 1] = data.x[:, 1] * self.std_pressure + self.mean_pressure  # Unnormalize pressure
        return data

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]
