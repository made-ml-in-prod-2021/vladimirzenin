import sys
sys.path.insert(0, '../scripts')

import pytest
from data_models import DataParams, SplittingParams, TrainParams
from model import model, save_model, load_model
from sklearn.linear_model import LogisticRegression


def data_params():
	return {"input_data_path": "test.csv",
			"output_model_path": "model.pkl",
			"metric_path": "metrics.json",
			"output_predict_path": "predict.csv"}


def splitting_params():
	return {"test_size": 0.5,
			"random_state": 1}


def train_params():
	return {"model_type": "LogisticRegression"}


def main():
	params = data_params()
	obj = DataParams(**params)
	assert (obj.input_data_path, obj.output_model_path, obj.metric_path, obj.output_predict_path) == (params['input_data_path'], params['output_model_path'], params['metric_path'], params['output_predict_path'])

	params = splitting_params()
	obj = SplittingParams(**params)
	assert (obj.test_size, obj.random_state) == (params['test_size'], params['random_state'])

	params = train_params()
	obj = TrainParams(**params)
	assert (obj.model_type)	== (params['model_type'])	
	
	work_model = model(obj)
	assert isinstance(work_model, LogisticRegression)
	
	path = "test_save_model.pkl"
	save_model(work_model, path)
	work2_model = load_model(path)
	assert type(work_model) == type(work2_model)



if __name__ == "__main__":
	main()

