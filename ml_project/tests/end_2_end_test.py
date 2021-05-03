import sys
import os
sys.path.insert(0, "../scripts")

import pytest
from data_models import read_data_params
from create_dataset import create_dataset
import click


@click.command()
@click.option("--config", default='../configs/config_test.yaml', help="path to yaml config.")
def main(config: str):
	data_params = read_data_params(config)
	data = create_dataset()
	data.to_csv(data_params.input_data_path, index=False, header=True)
	assert(os.path.exists(data_params.input_data_path))
	
	os.system(f"python ../scripts/train.py --config {config}")
	assert(os.path.exists(data_params.output_model_path))
	
	os.system(f"python ../scripts/predict.py --config {config}") 
	assert(os.path.exists(data_params.output_predict_path))


if __name__ == "__main__":
	main()

