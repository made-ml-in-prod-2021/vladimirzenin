import os
import pickle
import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import NoReturn


@click.command("train")
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--output_last_model_dir")
def train(input_dir: str, output_dir: str, output_last_model_dir: str) -> NoReturn:
    
    X = pd.read_csv(os.path.join(input_dir, "data_for_train.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target_for_train.csv"))
    
    model = LogisticRegression()
    model.fit(X, y)
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    os.makedirs(output_last_model_dir, exist_ok=True)
    path = os.path.join(output_last_model_dir, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    train()