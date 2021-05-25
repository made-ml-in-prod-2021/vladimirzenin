import pandas as pd
from random import randint, random


def get_columns() -> list:
	return ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]


def get_random_string() -> dict:
	return {"age": randint(15, 60),
			"sex": randint(0, 1),
			"cp": randint(0, 3),
			"trestbps": randint(100, 145),
			"chol": randint(131, 250),
			"fbs": randint(0, 1),
			"restecg": randint(0, 1),
			"thalach": randint(106, 195),
			"exang": randint(0, 1),
			"oldpeak": randint(0, 3) + random(),
			"slope": randint(0, 2),
			"ca": randint(0, 4),
			"thal": randint(1, 3),
			"target": randint(0, 1)}


def create_dataset() -> pd.DataFrame:
	answ = []
	for i in range(100):
		answ.append(get_random_string())
	return pd.DataFrame(answ, columns=get_columns())

