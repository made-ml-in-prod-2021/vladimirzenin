import sys
sys.path.insert(0, '../scripts')

import pytest
from data_models import TrainParams
from model import model, save_model, load_model, get_predict, get_evaluate_model
from sklearn.linear_model import LogisticRegression
from typing import NoReturn
from create_dataset import create_dataset
import pandas as pd


def train_params() -> NoReturn:
	return TrainParams(model_type="LogisticRegression")


@pytest.mark.maintest
def test_module_model_instance() -> NoReturn:
	obj = train_params()
	work_model = model(obj)
	assert isinstance(work_model, LogisticRegression)


@pytest.mark.maintest
def test_module_model_save_load() -> NoReturn:
	obj = train_params()
	work_model = model(obj)
	
	path = "test_save_model.pkl"
	save_model(work_model, path)
	work2_model = load_model(path)
	assert type(work_model) == type(work2_model)


@pytest.mark.maintest
def test_module() -> NoReturn:
	pretrain_model = load_model("../models/model.pkl")
	assert isinstance(pretrain_model, LogisticRegression)
	dataset = create_dataset()
	X = dataset.drop(columns=["target"])
	y = pd.DataFrame(dataset["target"])
	mae, mse = get_evaluate_model(pretrain_model, X, y)
	assert mae >= 0.0 and mae <= 1.0
	assert mse >= 0.0 and mse <= 1.0
	
	predict = get_predict(pretrain_model, X)
	assert predict[0] == 0 or predict[0] == 1
	assert len(predict) == len(X)


if __name__ == "__main__":
	test_module_model_instance()
	test_module_model_save_load()
	test_module_model_instance()

