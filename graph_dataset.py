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
    def __init__(self, batch_parquet_path: str, meta_parquet_path: str, k_neighbors: int = 8, n_rows: int = None):
        super().__init__(root=None, transform=None, pre_transform=None)

        self.batch_path = batch_parquet_path
        self.k_neighbors = k_neighbors

        if n_rows is not None:
            self.meta_df = pl.read_parquet(meta_parquet_path, n_rows=n_rows)
        else:
            self.meta_df = pl.read_parquet(meta_parquet_path)

        #self.meta_df = pl.read_parquet(meta_parquet_path)
        self.event_ids = self.meta_df.get_column("event_id").to_list()

        def __len__(self):
            return len(self.event_ids)
        
        def __getiitem__(self, idx):
            target_event_id = self.event_ids[idx]

            event_pulses_df = (
                               pl.scan_parquet(self.batch_path)
                               .filter(pl.col("event_id") == target_event_id)
                               .collect()
        )
            meta_row = self.meta_df.filter(pl.col("event_id") == target_event_id)
            azimuth = meta_row.get_column("azimuth")[0]
            zenith = meta_row.get_column("zenith")[0]

            graph = build_graph_for_event(event_pulses_df, azimuth, zenith)
            graph.event_id = target_event_id 
            return graph


if __name__ == "__main__":

    test_dataset = IceCubeGraphDataset(
        batch_parquet_path="data/raw/batch_1.parquet",
        meta_parquet_path="data/raw/train_meta.parquet",
        n_rows=10000 
    )
    
    graph_0 = test_dataset[0]
    
    graph_10 = test_dataset[10]