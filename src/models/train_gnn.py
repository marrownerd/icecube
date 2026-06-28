import os
import getpass
import torch
import yaml
import mlflow
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from src.data.graph_dataset import IceCubeGraphDataset

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class SimpleGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(5, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, 3)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        
        h_graph = global_mean_pool(h, batch)
        
        return self.out(h_graph)


def train_epoch(model, loader, optimizer, criterion):
    model.train() 
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        
        loss = criterion(out, batch.y)

        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval() # переводим модель в режим тестирования (выключает dropout и т.д.)
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            total_loss += loss.item() * batch.num_graphs
            
    return total_loss / len(loader.dataset)


def main():
    print("=== Запуск обучения Графовой Нейросети (GNN) ===")
    config = load_config("configs/params.yaml")
    
    REPO_OWNER = "marrownerd" 
    REPO_NAME = "icecube"
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
    
    if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("Введите DagsHub Token: ")
        
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("IceCube_GNN_Training")

    dataset = IceCubeGraphDataset(folder_path="data/processed/graphs/")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    hidden_size = 64
    model = SimpleGNN(hidden_channels=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = MSELoss() 
    
    epochs = 15
    print(f"Старт обучения на {epochs} эпох...")
    
    with mlflow.start_run(run_name="gcn_baseline"):
        mlflow.log_param("model_type", "GCN")
        mlflow.log_param("hidden_channels", hidden_size)
        mlflow.log_param("epochs", epochs)
        
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, val_loader, criterion)
            
            print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
            
            mlflow.log_metric("train_mse", train_loss, step=epoch)
            mlflow.log_metric("val_mse", val_loss, step=epoch)
            
        mlflow.pytorch.log_model(model, artifact_path="gnn_models")
        

if __name__ == "__main__":
    main()