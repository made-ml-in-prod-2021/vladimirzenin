from sklearn.linear_model import LogisticRegression
from data_models import TrainParams, DataParams
import logging
import pickle
import os.path
import json
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import NoReturn
from numpy import array


def model(train_params: TrainParams) -> LogisticRegression:
	if train_params.model_type == "LogisticRegression":
		return LogisticRegression(max_iter=2000)
	else:
		logging.error(f'Model {train_params.model_type} not implemented!')
		return None


def get_evaluate_model(work_model: LogisticRegression, X_test: DataFrame, y_test: DataFrame) -> (float, float):
	logging.info('start evaluate model')
	predict = work_model.predict(X_test)
	mae = mean_absolute_error(y_test, predict)
	mse = mean_squared_error(y_test, predict)
	return mae, mse


def evaluate_model(work_model: LogisticRegression, X_test: DataFrame, y_test: DataFrame, data_params: DataParams) -> NoReturn:
	mae, mse = get_evaluate_model(work_model, X_test, y_test)
	with open(data_params.metric_path, "w") as stream:
		json.dump({"mae": mae, "mse": mse}, stream)


def get_predict(work_model: LogisticRegression, X: DataFrame) -> array:
	logging.info('start prediction model')
	predict = work_model.predict(X)
	return predict


def predict(work_model: LogisticRegression, X: DataFrame, data_params: DataParams) -> NoReturn:
	predict = get_predict(work_model, X)
	DataFrame(predict).to_csv(data_params.output_predict_path, index=False, header=False)


def simple_predict(work_model: LogisticRegression, X: DataFrame) -> list:
	predict = get_predict(work_model, X)
	return list(predict)


def save_model(work_model: LogisticRegression, path: str) -> NoReturn:
	logging.info('start saving model')
	with open(path, "wb") as stream:
		pickle.dump(work_model, stream)


def load_model(path: str) -> LogisticRegression:
	logging.info('start loading model')
	if os.path.exists(path):
		with open(path, "rb") as stream:
			work_model = pickle.load(stream)
		return work_model
	else:
		logging.error('Model file not exist!')
		return None

