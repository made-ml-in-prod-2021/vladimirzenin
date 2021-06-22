import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "01-first-train",
        default_args=default_args,
        schedule_interval="@once",
        start_date=days_ago(0),
) as dag:
    data_download = DockerOperator(
        image="airflow-download",
        command="--dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=["E:/ML/MADE_2020_Learn/2. ml in prod/hw0/airflow/data:/data"]
    )
    
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/proc_data/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=["E:/ML/MADE_2020_Learn/2. ml in prod/hw0/airflow/data:/data"]
    )
    
    train = DockerOperator(
        image="airflow-train",
        command="--input_dir /data/proc_data/{{ ds }} --output_dir /data/models/{{ ds }} --output_last_model_dir /data/last_model/",
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=["E:/ML/MADE_2020_Learn/2. ml in prod/hw0/airflow/data:/data"]
    )
    
    valid = DockerOperator(
        image="airflow-valid",
        command="--input_data_dir /data/proc_data/{{ ds }} --input_model_dir /data/models/{{ ds }} --output_dir /data/metrics/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-valid",
        do_xcom_push=False,
        volumes=["E:/ML/MADE_2020_Learn/2. ml in prod/hw0/airflow/data:/data"]
    )

    data_download >> preprocess >> train >> valid
