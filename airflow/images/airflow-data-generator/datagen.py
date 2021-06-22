import os
import click
import pandas as pd
from random import uniform, randint


def get_columns() -> list:
    return ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline", "target"]


def get_random_string() -> dict:
    return {"alcohol": uniform(11.03, 14.83),
            "malic_acid": uniform(0.74, 5.8),
            "ash": uniform(1.36, 3.23),
            "alcalinity_of_ash": uniform(10.6, 30.0),
            "magnesium": uniform(70.0, 162.0),
            "total_phenols": uniform(0.98, 3.88),
            "flavanoids": uniform(0.34, 5.08),
            "nonflavanoid_phenols": uniform(0.13, 0.66),
            "proanthocyanins": uniform(0.41, 3.58),
            "color_intensity": uniform(1.28, 13.0),
            "hue": uniform(0.48, 1.71),
            "od280/od315_of_diluted_wines": uniform(1.27, 4.0),
            "proline": uniform(278.0, 1680.0),
            "target": randint(0, 2)}


def create_dataset() -> pd.DataFrame:
    answ = []
    for i in range(50):
        answ.append(get_random_string())
    return pd.DataFrame(answ, columns=get_columns())


@click.command("datagen")
@click.option("--dir")
def datagen(dir: str):
    dataset = create_dataset()
    y = dataset['target']
    X = dataset.drop(columns='target')

    os.makedirs(dir, exist_ok=True)
    X.to_csv(os.path.join(dir, "data.csv"), index=False)
    y.to_csv(os.path.join(dir, "target.csv"), index=False)


if __name__ == '__main__':
    datagen()
