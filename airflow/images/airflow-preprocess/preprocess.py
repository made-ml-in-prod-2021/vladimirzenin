import os
import pandas as pd
import click


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    
    X["ash"] += 1 # Просто некоторая коррекция.
    X["alcalinity_of_ash"] -= 2
    
    # Разделение на трейн/валидацию и выделение данные и таргета.
    
    bord = int(len(X) * 0.8)
    X_train = X[:bord]
    X_test = X[bord:]
    y_train = y[:bord]
    y_test = y[bord:]

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "data_for_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "target_for_train.csv"), index=False)

    X_test.to_csv(os.path.join(output_dir, "data_for_valid.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "target_for_valid.csv"), index=False)

if __name__ == '__main__':
    preprocess()