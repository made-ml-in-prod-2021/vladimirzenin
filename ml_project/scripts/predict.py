import logging
from get_train_test import get_predict_data, get_df_from_json
from data_models import read_data_params, JsonArr
from model import predict, load_model, simple_predict
import click


@click.command()
@click.option("--config", default='../configs/config.yaml', help="path to yaml config.")
def main(config: str):
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


def get_predict_json(input_data: JsonArr) -> str:
	config = "../../ml_project/configs/config.yaml"
	output_model_path = "../../ml_project/models/model.pkl"
	#data_params = read_data_params(config)
	#if data_params is None:
	#	return	None
	
	X = get_df_from_json(input_data)
	if X is None:
		return None
	
	work_model = load_model(output_model_path)
	if work_model is None:
		return
	answ = simple_predict(work_model, X)
	return ','.join([str(n) for n in answ])


if __name__ == "__main__":
	main()

