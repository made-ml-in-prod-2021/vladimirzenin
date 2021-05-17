import logging
from get_train_test import get_train_test
from data_models import read_data_params, read_splitting_params, read_train_params
from model import model, save_model, evaluate_model
import click


@click.command()
@click.option("--config", default='../configs/config.yaml', help="path to yaml config.")
def main(config: str):
	logging.info('start')
	data_params = read_data_params(config)
	splitting_params = read_splitting_params(config)
	train_params = read_train_params(config)
	if data_params is None or splitting_params is None or train_params is None:
		return	
	logging.info('parameters read')
	
	X_train, X_test, y_train, y_test = get_train_test(data_params, splitting_params)
	if X_train is None:
		return
	logging.info('received data')
	
	work_model = model(train_params)
	if work_model is None:
		return
	logging.info('model created')
	
	work_model.fit(X_train, y_train)
	logging.info('model trained')
	
	evaluate_model(work_model, X_test, y_test, data_params)
	logging.info('model evaluated')
	
	save_model(work_model, data_params.output_model_path)


#@click.command(name="train_pipeline")
#@click.argument("config")
#def train_pipeline_command(config: str):
#	main(config)


if __name__ == "__main__":
	main()

