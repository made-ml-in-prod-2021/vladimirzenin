FROM python:3.9-slim

EXPOSE 8050

LABEL name=ml_in_prod \
	hw=2 \
	student=zenin

# requirements меняются реже всего, поэтому их и их установку - в начало.
COPY ml_project/scripts/requirements.txt /scripts/requirements.txt
COPY online_inference/server/requirements.txt /server/requirements.txt

# Объединено в одну команду с целью сокращения количества слоев.
RUN pip install -r /scripts/requirements.txt --no-cache-dir \
	&& pip install -r /server/requirements.txt --no-cache-dir

RUN groupadd -r hw_usr && useradd --no-log-init -r -g hw_usr hw_usr

USER hw_usr

# В остальных файлах изменения могут быть чаще.
COPY ml_project/configs /configs
COPY ml_project/models/model.pkl /models/model.pkl

# Наибольший шанс на изменения в файлах скриптов.
COPY ml_project/scripts /scripts
COPY online_inference/server /server

WORKDIR server

CMD ["sh", "-c", "python3 m_test.py && python3 m_server.py"]
