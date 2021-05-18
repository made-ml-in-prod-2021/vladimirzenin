from fastapi.testclient import TestClient
from m_server import app
import json

client = TestClient(app)


def test_predict_single():
	f = open('valid_query_item.json', 'r')
	dct_txt = json.load(f)
	response = client.post("/predict_single/", headers={"accept": "application/json", "Content-Type": "application/json"}, json=dct_txt)
	assert response.status_code == 200
	assert response.json() == "1"
	
	print('PASS Test predict_single for valid json')


def test_predict_single_not_valid():
	f = open('not_valid_query_item.json', 'r')
	dct_txt = json.load(f)
	response = client.post("/predict_single/", headers={"accept": "application/json", "Content-Type": "application/json"}, json=dct_txt)
	assert response.status_code == 400
	
	print('PASS Test predict_single for not valid json')


def test_predict_batch():
	f = open('valid_query_batch.json', 'r')
	dct_txt = json.load(f)
	response = client.post("/predict_batch/", headers={"accept": "application/json", "Content-Type": "application/json"}, json=dct_txt)
	assert response.status_code == 200
	assert response.json() == "1,1,1"
	
	print('PASS Test predict_single for valid batch json')


def test_predict_batch_not_valid():
	f = open('not_valid_query_batch.json', 'r')
	dct_txt = json.load(f)
	response = client.post("/predict_batch/", headers={"accept": "application/json", "Content-Type": "application/json"}, json=dct_txt)
	assert response.status_code == 400
	
	print('PASS Test predict_single for not valid batch json')


test_predict_single()
test_predict_single_not_valid()
test_predict_batch()
test_predict_batch_not_valid()
