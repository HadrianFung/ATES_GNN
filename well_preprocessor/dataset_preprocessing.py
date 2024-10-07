# name : Hadrian Fung
# Github username : gems-hf923
# dataset_preprocessing.py

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from scipy.spatial import KDTree
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

def read_vtu(file_path):
    import meshio
    mesh = meshio.read(file_path)
    return mesh

def generate_injection_sequence(total_timesteps=240):
    sequence = [1] * 4  # Initial injection in Well 1 for the first 4 timesteps
    cycle = [0] * 4 + [2] * 8 + [0] * 4 + [1] * 8  # Define the repeating cycle

    remaining_timesteps = total_timesteps - 4  # Subtract the initial 4 timesteps
    num_cycles = remaining_timesteps // len(cycle)  # Number of complete cycles that fit into the remaining timesteps

    for _ in range(num_cycles):
        sequence.extend(cycle)

    remainder = remaining_timesteps % len(cycle)  # Add remaining part if total_timesteps isn't a multiple of cycle length
    if remainder > 0:
        sequence.extend(cycle[:remainder])

    return sequence

def read_csv(csv_path, scenario_id):
    df = pd.read_csv(csv_path)
    params = df[df['scenario'] == scenario_id].iloc[0]
    return params

def create_graph(points, node_data, edge_data, scenario_id, timestep, injection_sequence, csv_path='data_subset6_graph/raw/scenariodatabase_ATES.csv'):
    # Read the scenario-specific parameters
    params = read_csv(csv_path, scenario_id)

    # Define the layer boundaries based on the provided parameters
    underburden_depth = -params['Thickness_Underburden [m]']
    aquifer_depth = underburden_depth - params['Thickness_Aquifer [m]']
    overburden_depth = aquifer_depth - params['Thickness_Overburden [m]']

    # Get the Kz values for the layers
    underburden_kz = params['Underburden_Kz [mD]']
    aquifer_kz = params['Aquifer_Kz [mD]']
    overburden_kz = params['Overburden_Kz [mD]']

    # Define the well parameters
    well1_depth = params['Well1_depth [m]']
    well2_depth = params['Well2_depth [m]']
    well1_screen_top = well1_depth
    well1_screen_bottom = well1_depth + params['Well1_screenlength [m]']
    well2_screen_top = well2_depth
    well2_screen_bottom = well2_depth + params['Well2_screenlength [m]']
    well_spacing = params['Wellspacing [m]'] / 2.0  # Half spacing for positive and negative coordinates

    # Extract the x, y, z coordinates from the points array
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    # Vectorized assignment of Kz based on depth
    kz = np.where(
        z_coords >= underburden_depth, underburden_kz,
        np.where(z_coords >= aquifer_depth, aquifer_kz, overburden_kz)
    )

    # Determine injection for the current timestep
    injection_encoder = injection_sequence[timestep - 1]  # Subtract 1 because sequence is 0-indexed
    next_injection_encoder = injection_sequence[timestep] if timestep < len(injection_sequence) else 0

    # Vectorized assignment of injection status for current and next timestep
    injection = np.zeros(len(points))
    next_injection = np.zeros(len(points))

    if injection_encoder == 1:
        injection = np.where(
            (z_coords >= well1_screen_top) & (z_coords <= well1_screen_bottom) &
            ((np.abs(x_coords - well_spacing) < 1e-3) & (np.abs(y_coords) < 1e-3)),
            1, 0
        )
    elif injection_encoder == 2:
        injection = np.where(
            (z_coords >= well2_screen_top) & (z_coords <= well2_screen_bottom) &
            ((np.abs(x_coords + well_spacing) < 1e-3) & (np.abs(y_coords) < 1e-3)),
            1, 0
        )

    if next_injection_encoder == 1:
        next_injection = np.where(
            (z_coords >= well1_screen_top) & (z_coords <= well1_screen_bottom) &
            ((np.abs(x_coords - well_spacing) < 1e-3) & (np.abs(y_coords) < 1e-3)),
            1, 0
        )
    elif next_injection_encoder == 2:
        next_injection = np.where(
            (z_coords >= well2_screen_top) & (z_coords <= well2_screen_bottom) &
            ((np.abs(x_coords + well_spacing) < 1e-3) & (np.abs(y_coords) < 1e-3)),
            1, 0
        )

    # Stack all the features together into a single array, ensuring coordinates are included
    node_features = np.column_stack([
        node_data['phase1::Temperature'], 
        node_data['phase1::Pressure'], 
        x_coords, 
        y_coords, 
        z_coords, 
        kz, 
        injection, 
        next_injection
    ])

    node_features = torch.tensor(node_features, dtype=torch.float)

    # Process edges as before
    edge_index = []
    edge_attr = []
    for edge, attrs in edge_data.items():
        edge_index.append([edge[0], edge[1]])
        edge_attr.append(attrs)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    # Create the graph data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data

def project_cell_data_to_edges(cells, points):
    
    edge_data = {}

    for cell in cells:
        # Generate all edge pairs within the tetrahedron (4 nodes)
        edges = [(cell[i], cell[j]) for i in range(4) for j in range(i + 1, 4)]
        for edge in edges:
            # Calculate the Euclidean distance between the points
            distance = np.linalg.norm(points[edge[0]] - points[edge[1]])
            
            # Only store the distance for one direction in an undirected graph
            if edge not in edge_data:
                edge_data[edge] = distance

    return edge_data  

def project_graph_to_previous_mesh(graph_t, graph_t_plus_1):
    from scipy.spatial import KDTree

    # Extract the x, y, z coordinates from the node features of the current and next timestep graphs
    coords_t = graph_t.x[:, 2:5].numpy()  # Coordinates in graph at time t (x, y, z are now indices 2, 3, 4)
    coords_t_plus_1 = graph_t_plus_1.x[:, 2:5].numpy()  # Coordinates in graph at time t+1
    features_t_plus_1 = graph_t_plus_1.x[:, :2].numpy()  # Assuming the first two columns are Temperature and Pressure

    # Use KDTree to map points from t+1 to t
    tree = KDTree(coords_t_plus_1)
    _, idx = tree.query(coords_t)

    # Interpolate Temperature and Pressure features from t+1 to t
    interpolated_features = features_t_plus_1[idx]

    # Assign interpolated features as the target (y) in the current graph
    graph_t.y = torch.tensor(interpolated_features, dtype=torch.float)

    return graph_t

def project_graph_to_next_mesh(graph_t, graph_t_plus_1):

    coords_t = graph_t.x[:, 2:5].numpy()
    coords_t_plus_1 = graph_t_plus_1.x[:, 2:5].numpy()
    features_t = graph_t.x[:, :2].numpy()

    # Use KDTree to map points from t to t+1
    tree = KDTree(coords_t)
    _, idx = tree.query(coords_t_plus_1)

    # Interpolate Temperature and Pressure features from t+1 to t
    interpolated_features = features_t[idx]

    projected_graph_t = graph_t_plus_1.clone()
    projected_graph_t.y = projected_graph_t.x[:,:2].clone()
    projected_graph_t.x[:,:2] = torch.tensor(interpolated_features, dtype=torch.float)

    # Translate the well injection phase to the new mesh
    inj_t = graph_t.x[:,-2].numpy()
    inj_t_plus_1 = graph_t.x[:,-1].numpy()

    inj_well_t = np.where(inj_t == 1)[0]
    inj_well_t_plus_1 = np.where(inj_t_plus_1 == 1)[0]

    if len(inj_well_t) != 0:
        inj_well_coord_t = coords_t[inj_well_t, :2][0]
        inj_well_screen_t = coords_t[inj_well_t, 2]
        inj_well_well_depth_range_t = (min(inj_well_screen_t), max(inj_well_screen_t))
        proj_inj_well_t = np.where(
            (coords_t_plus_1[:,0] == inj_well_coord_t[0]) & (coords_t_plus_1[:,1] == inj_well_coord_t[1]) & 
            (coords_t_plus_1[:,2] >= inj_well_well_depth_range_t[0]) & (coords_t_plus_1[:,2] <= inj_well_well_depth_range_t[1]),
            1, 0
        )
    else:
        proj_inj_well_t = np.zeros_like(projected_graph_t.x[:,-2])

    projected_graph_t.x[:,-1] = graph_t_plus_1.x[:,-2].clone()
    projected_graph_t.x[:,-2] = torch.tensor(proj_inj_well_t, dtype=torch.float)

    return projected_graph_t

class TemporalGraphDataset(InMemoryDataset):
    def __init__(self, root, stats=None, force_reprocess=False, max_t=20, projection_method='backward'):
        self.max_t = max_t
        self.force_reprocess = force_reprocess
        self.stats = stats
        self.projection_method = projection_method
        super(TemporalGraphDataset, self).__init__(root)
        if self.force_reprocess or not os.path.exists(self.processed_paths[0]):
            self.process()
        self._data_list = torch.load(self.processed_paths[0])
        
        if self.stats is None:
            self.normalize_data()  # Compute stats and normalize
        else:
            self.mean_temp, self.std_temp, self.mean_pressure, self.std_pressure = self.stats
            self.apply_normalization()  # Use provided stats to normalize

    @property
    def raw_file_names(self):
        scenarios = os.listdir(self.raw_dir)
        all_files = []
        for scenario in scenarios:
            scenario_path = os.path.join(self.raw_dir, scenario)
            if os.path.isdir(scenario_path) and not scenario.startswith('.'):
                for timestep in range(1, self.max_t + 1):
                    filename = f"z_ATES_{scenario}_{timestep}.vtu"
                    all_files.append(filename)
        return all_files

    @property
    def processed_file_names(self):
        return [f'data_T{self.max_t}.pt']

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def process(self):
        data_list = []

        scenarios = [os.path.join(self.raw_dir, scenario) for scenario in os.listdir(self.raw_dir) if not scenario.startswith('.') and os.path.isdir(os.path.join(self.raw_dir, scenario))]

        # Generate the injection sequence once and reuse it
        injection_sequence = generate_injection_sequence()

        for scenario in scenarios:
            scenario_name = os.path.basename(scenario)
            scenario_id = int(''.join(filter(str.isdigit, scenario_name)))  # Extract numbers from the scenario name

            scenario_graphs = []
            for timestep in range(1, self.max_t + 1):
                print(f"Processing scenario {scenario} at timestep {timestep}...")
                file_path = os.path.join(scenario, f"z_ATES_{scenario_name}_{timestep}.vtu")
                if os.path.exists(file_path):
                    vtu_0 = read_vtu(file_path)
                    node_data = vtu_0.point_data
                    cell_data = vtu_0.cell_data_dict
                    cells = np.vstack([cell.data for cell in vtu_0.cells if cell.type == "tetra"])
                    edge_data = project_cell_data_to_edges(cells, vtu_0.points)  # Modify this function as necessary
                    graph = create_graph(vtu_0.points, node_data, edge_data, scenario_id, timestep, injection_sequence)
                    scenario_graphs.append(graph)

            for t in range(len(scenario_graphs) - 1):
                graph_t = scenario_graphs[t]
                graph_t_plus_1 = scenario_graphs[t + 1]
                if self.projection_method == 'backward':
                    projected_graph = project_graph_to_previous_mesh(graph_t, graph_t_plus_1)
                elif self.projection_method == 'forward':
                    projected_graph = project_graph_to_next_mesh(graph_t, graph_t_plus_1)
                data_list.append(projected_graph)

        torch.save(data_list, self.processed_paths[0])
        self._data_list = data_list

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
                data.y[:, 0] = (data.y[:, 0] - self.mean_temp) / self.std_temp  # Normalize target temperature
                data.y[:, 1] = (data.y[:, 1] - self.mean_pressure) / self.std_pressure  # Normalize target pressure

        # (Optional) Logging the normalized values for verification
        norm_temps = [data.x[:, 0] for data in self._data_list if data is not None]
        norm_temps = torch.cat(norm_temps, dim=0)
        print(f"Normalized temperatures - Mean: {norm_temps.mean()}, Std: {norm_temps.std()}, Min: {norm_temps.min()}, Max: {norm_temps.max()}")
        
        norm_pressures = [data.x[:, 1] for data in self._data_list if data is not None]
        norm_pressures = torch.cat(norm_pressures, dim=0)
        print(f"Normalized pressures - Mean: {norm_pressures.mean()}, Std: {norm_pressures.std()}, Min: {norm_pressures.min()}, Max: {norm_pressures.max()}")

    def unnormalize(self, data):
        data.x[:, 0] = data.x[:, 0] * self.std_temp + self.mean_temp  # Unnormalize temperature
        data.x[:, 1] = data.x[:, 1] * self.std_pressure + self.mean_pressure  # Unnormalize pressure
        data.y[:, 0] = data.y[:, 0] * self.std_temp + self.mean_temp  # Unnormalize target temperature
        data.y[:, 1] = data.y[:, 1] * self.std_pressure + self.mean_pressure  # Unnormalize target pressure
        return data

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

if __name__ == '__main__':
    pass