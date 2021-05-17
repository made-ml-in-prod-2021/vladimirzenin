FROM python:3.6

COPY ml_project/scripts /scripts
RUN pip install -r /scripts/requirements.txt

COPY online_inference/docker/fast_server.py /fast_serv
RUN pip install unicorn fastapi

COPY ml_project/configs /configs
COPY ml_project/data /data

RUN mkdir -p /models

WORKDIR .

#CMD ["python3", "scripts/train.py", "--config", "../configs/config.yaml"]
CMD ["uvicorn", "main:app", " --reload"]
