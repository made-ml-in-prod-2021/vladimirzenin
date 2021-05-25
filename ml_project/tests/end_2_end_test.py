import sys
import os
sys.path.insert(0, "../scripts")

import pytest
from create_dataset import create_dataset
from data_models import read_data_params
import click
from typing import NoReturn


def end_to_end(config: str) -> NoReturn:
	data_params = read_data_params(config)
	data = create_dataset()
	data.to_csv(data_params.input_data_path, index=False, header=True)
	assert(os.path.exists(data_params.input_data_path))
	
	os.system(f"python ../scripts/train.py --config {config}")
	assert(os.path.exists(data_params.output_model_path))
	
	os.system(f"python ../scripts/predict.py --config {config}") 
	assert(os.path.exists(data_params.output_predict_path))


@click.command()
@click.option("--config", help="path to yaml config.")
def console_end_to_end(config: str) -> NoReturn:
	end_to_end(config)


@pytest.mark.maintest
def test_end_to_end() -> NoReturn:
	default_path = '../configs/config_test.yaml'
	end_to_end(default_path)


if __name__ == "__main__":
	console_end_to_end()

