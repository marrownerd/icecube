import os 
import torch
import numpy as np
import polars as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import knn_graph

def build_graph_for_event(event_df: pl.DataFrame, target_azimuth: float, target_zenith: float, k_neighbors: int) -> Dataset:
    
    x_tensor = torch.tensor(event_df.select(['x', 'y', 'z', 'time', 'charge']).to_numpy(), dtype=torch.float)

    
    pos_tensor = torch.tensor(event_df.select(['x', 'y', 'z']).to_numpy(), dtype=torch.float32)
    
    edge_index = knn_graph(pos_tensor, k=8)
    
    dir_x = np.sin(target_zenith) * np.cos(target_azimuth)
    dir_y = np.sin(target_zenith) * np.sin(target_azimuth)
    dir_z = np.cos(target_zenith)
    
    y_tensor = torch.tensor([[dir_x, dir_y, dir_z]], dtype=torch.float32)
    
    graph = Data(x=x_tensor, edge_index=edge_index, pos=pos_tensor, y=y_tensor)
    
    return graph

class IceCubeGraphDataset(Dataset):
    def __init__(self, folder_path: str):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.folder_path = folder_path
        
        if os.path.exists(folder_path):
            self.file_names = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
        else:
            self.file_names = []
            
    def len(self):
        return len(self.file_names)
        
    def get(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        graph = torch.load(file_path, weights_only=False) 
        return graph

if __name__ == "__main__":

    test_dataset = IceCubeGraphDataset(
        batch_parquet_path="data/raw/batch_1.parquet",
        meta_parquet_path="data/raw/train_meta.parquet",
        n_rows=10000 
    )
    
    graph_0 = test_dataset[0]
    
    graph_10 = test_dataset[10]