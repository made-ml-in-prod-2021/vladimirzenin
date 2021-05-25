import logging
import yaml
from dataclasses import dataclass


@dataclass
class DataParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    output_predict_path: str


@dataclass
class SplittingParams:
    test_size: float
    random_state: int


@dataclass
class TrainParams:
    model_type: str


def read_data_params(path: str) -> DataParams:
	with open(path, 'r') as stream:
		try:
			params = yaml.safe_load(stream)
			return DataParams(**params['data_params'])
		except yaml.YAMLError as exc:
			logging.error('error loading params:')
			logging.error(exc)


def read_splitting_params(path: str) -> SplittingParams:
	with open(path, 'r') as stream:
		try:
			params = yaml.safe_load(stream)
			return SplittingParams(**params['splitting_params'])
		except yaml.YAMLError as exc:
			logging.error('error loading params:')
			logging.error(exc)


def read_train_params(path: str) -> TrainParams:
	with open(path, 'r') as stream:
		try:
			params = yaml.safe_load(stream)
			return TrainParams(**params['train_params'])
		except yaml.YAMLError as exc:
			logging.error('error loading params:')
			logging.error(exc)

