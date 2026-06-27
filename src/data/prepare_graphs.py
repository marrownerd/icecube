import os
import polars as pl
import torch
import yaml
from tqdm import tqdm 
from graph_dataset import build_graph_for_event

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config("configs/params.yaml")
    
    output_dir = "data/processed/graphs/"
    os.makedirs(output_dir, exist_ok=True)
    
    geom_df = pl.read_csv("data/raw/sensor_geometry.csv")
    
    raw_pulses = pl.read_parquet(config["data"]["raw_batch"], n_rows=100000)
    
    pulses_with_geom = raw_pulses.join(geom_df, on="sensor_id", how="left")
    
    events_list = pulses_with_geom.partition_by("event_id", as_dict=False)
    
    unique_ids = [event["event_id"][0] for event in events_list]
    
    meta_df = (
        pl.scan_parquet(config["data"]["meta"])
        .filter(pl.col("event_id").is_in(unique_ids))
        .collect()
    )
    
    meta_dict = {
        row["event_id"]: (row["azimuth"], row["zenith"]) 
        for row in meta_df.to_dicts()
    }

    for event_df in tqdm(events_list):
        event_id = event_df["event_id"][0]
        
        if event_id in meta_dict:
            azimuth, zenith = meta_dict[event_id]
        else:
            continue 

        graph = build_graph_for_event(event_df, azimuth, zenith, k_neighbors=8)
        
        save_path = os.path.join(output_dir, f"event_{event_id}.pt")
        torch.save(graph, save_path)


if __name__ == "__main__":
    main()