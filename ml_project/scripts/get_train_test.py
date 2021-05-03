import logging
import os.path
from data_models import DataParams, SplittingParams
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def get_train_test(data_params: DataParams, splitting_params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if os.path.exists(data_params.input_data_path):
		data = pd.read_csv(data_params.input_data_path)
		X = data.drop('target', axis=1)
		y = data['target']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitting_params.test_size, random_state=splitting_params.random_state)
		return X_train, X_test, y_train, y_test
	else:
		logging.error('file not exist!')
		return None, None, None, None


def get_predict_data(data_params: DataParams) -> pd.DataFrame:
	if os.path.exists(data_params.input_data_path):
		data = pd.read_csv(data_params.input_data_path)
		X = data.drop('target', axis=1)
		return X
	else:
		logging.error('file not exist!')
		return None

