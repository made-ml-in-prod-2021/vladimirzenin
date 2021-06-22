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
        "02-datagen",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    data_generator = DockerOperator(
        image="airflow-data-generator",
        command="--dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-data-generator",
        do_xcom_push=False,
        volumes=["E:/ML/MADE_2020_Learn/2. ml in prod/hw0/airflow/data:/data"]
    )

    data_generator
