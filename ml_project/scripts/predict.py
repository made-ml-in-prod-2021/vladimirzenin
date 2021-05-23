import logging
from get_train_test import get_predict_data
from data_models import read_data_params
from model import predict, load_model
import click
from typing import NoReturn


@click.command()
@click.option("--config", help="path to yaml config.")
def main(config: str) -> NoReturn:
	logging.info('start')
	data_params = read_data_params(config)
	if data_params is None:
		return	
	logging.info('parameters read')
	
	X = get_predict_data(data_params)
	if X is None:
		return
	logging.info('received data')
	
	work_model = load_model(data_params.output_model_path)
	if work_model is None:
		return
	logging.info('model created')
	
	predict(work_model, X, data_params)
	logging.info('model predicted')


if __name__ == "__main__":
	main()

