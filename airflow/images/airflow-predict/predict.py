import os
import pandas as pd
import click
import pickle
from typing import NoReturn


@click.command("predict")
@click.option("--input_data_dir")
@click.option("--input_model_dir")
@click.option("--output_dir")
def predict(input_data_dir: str, input_model_dir : str, output_dir: str) -> NoReturn:
    
    X = pd.read_csv(os.path.join(input_data_dir, "data_for_valid.csv"))
    y = pd.read_csv(os.path.join(input_data_dir, "target_for_valid.csv"))
    
    path = os.path.join(input_model_dir, "model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    pred = model.predict(X)
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(pred).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()