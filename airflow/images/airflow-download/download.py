import os

import click
from sklearn.datasets import load_wine


@click.command("download")
@click.option("--dir")
def download(dir: str):
    X, y = load_wine(return_X_y=True, as_frame=True)

    os.makedirs(dir, exist_ok=True)
    X.to_csv(os.path.join(dir, "data.csv"), index=False)
    y.to_csv(os.path.join(dir, "target.csv"), index=False)


if __name__ == '__main__':
    download()