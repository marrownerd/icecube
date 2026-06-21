import polars as pl
import yaml
import mlflow
import os
import getpass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    print("обучение модли")
    config = load_config("configs/params.yaml")
    
    REPO_OWNER = "marrownerd" 
    REPO_NAME = "icecube"
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
    
    if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("Введите DagsHub Token: ")
        
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("IceCube_MVP_Training")

    data_path = config["data"]["processed_features"]
    df = pl.read_parquet(data_path)
    
    df_pandas = df.to_pandas()

    X = df_pandas[["pulse_count", "total_charge"]]
    y = df_pandas[["azimuth", "zenith"]]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    

    alpha_param = 1.0 
    
    with mlflow.start_run(run_name="ridge_regression_baseline"):
        
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha_param)
        
        # - Обучи её на train выборке: model.fit(...)
        # - Сделай предсказание на val выборке: preds = model.predict(...)
        model = Ridge(alpha=alpha_param) 

        model.fit(X_train, y_train)
    
        preds = model.predict(X_val)
        
        val_mse = mean_squared_error(y_val, preds)
        
        print(f"Validation MSE: {val_mse:.4f}")
        
        mlflow.log_metric("val_mse", val_mse)
        
        mlflow.sklearn.log_model(model, artifact_path="models")

if __name__ == "__main__":
    main()