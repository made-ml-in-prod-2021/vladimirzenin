import os
import pickle
import click
import pandas as pd
from typing import NoReturn
from sklearn.metrics import mean_absolute_error
import json


@click.command("valid")
@click.option("--input_data_dir")
@click.option("--input_model_dir")
@click.option("--output_dir")
def valid(input_data_dir: str, input_model_dir : str, output_dir: str) -> NoReturn:
    
    X = pd.read_csv(os.path.join(input_data_dir, "data_for_valid.csv"))
    y = pd.read_csv(os.path.join(input_data_dir, "target_for_valid.csv"))
    
    path = os.path.join(input_model_dir, "model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump({"mae": mae}, f)

if __name__ == '__main__':
    valid()