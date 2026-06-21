import polars as pl
import yaml
# import os

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config("configs/params.yaml")
    
    raw_batch_path = config["data"]["raw_batch"]
    meta_path = config["data"]["meta"]
    output_path = config["data"]["processed_features"]
    sample_rows = config["data"]["sample_rows"] 
    
    print(f"1: агрегация первых строк из батча")
    
    features_df = (
        pl.scan_parquet(raw_batch_path, n_rows=sample_rows)
        .group_by("event_id")
        .agg(
            pl.len().alias("pulse_count"),
            pl.col("charge").sum().alias("total_charge")
        )
        .collect()
    )
    
    target_ids = features_df.get_column("event_id")

    print("2: фильтр метаданных на диске")
    
    meta_df = (
        pl.scan_parquet(meta_path)
        .filter(pl.col("event_id").is_in(target_ids))
        .collect()
    )
    
    print("3: left join по ключу event id")
    final_df = features_df.join(meta_df, on="event_id", how="left")
    
    final_df.write_parquet(output_path)
    config = load_config("configs/params.yaml")
    
    raw_batch_path = config["data"]["raw_batch"]
    meta_path = config["data"]["meta"]
    output_path = config["data"]["processed_features"]
    
    print("кусочек метаданных")
    meta_df = pl.read_parquet(meta_path, n_rows=50000)
    
    target_ids = meta_df.get_column("event_id")
    
    print("читаем и фильтруем фильтруем батч пульсов")
    features_df = (
        pl.scan_parquet(raw_batch_path)
        .filter(pl.col("event_id").is_in(target_ids))
        .group_by("event_id")
        .agg(
            pl.len().alias("pulse_count"),
            pl.col("charge").sum().alias("total_charge")
        )
        .collect()
    )
    
    print("объединяем")
    final_df = features_df.join(meta_df, on="event_id", how="left")
    
    print(f"Сохраняем в {output_path}...")
    final_df.write_parquet(output_path)

if __name__ == "__main__":
    main()