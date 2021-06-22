# airflow-examples
код для пары Data Pipelines

чтобы развернуть airflow, предварительно собрав контейнеры
~~~
# для корректной работы с переменными, созданными из UI
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build
~~~
Ссылка на документацию по docker compose up
  
https://docs.docker.com/compose/reference/up/
  
## Домашнее задание 3
-----------
### Содержимое проекта    
  
Образы:  
1. airflow-docker - задает базовый образ airflow-python  
2. airflow-ml-base - задает базовый образ python, устанавливает нужные библиотеки для работы тренировки/предсказания  
3. airflow-download (от airflow-ml-base) - скачивает датасет wine (178 строк, 13 колонок, 3 класса)  
4. airflow-data-generator (от airflow-ml-base) - формирует датасет в объеме 50 строк в формате wine  
5. airflow-preprocess (от airflow-ml-base) - считывает скаченные/сгенерированные данные, производит препроцессинг данных, разделяет на трейн/тест.  
6. airflow-train (от airflow-ml-base) - считывает подготовленные данные, тренирует модель, записывает дамп модели в файл  
7. airflow-valid (от airflow-ml-base) - считывает подготовленные тестовые данные, рассчитывает метрику mean_absolute_error, записывает ее в файл  
8. airflow-predict (от airflow-ml-base) - считывает data.csv, строит предсказание, записывает в файл  
     
DAG:  
1. 01_first_train - первоначальная тренировка модели на датасете wine  
2. 02_data_generator - входящие данные (из генератора данных)  
3. 03_weekly_train - еженедельное получение новых данных и тренировка модели на них  
4. 04_daily_predict - ежедневное предсказание на последней доступной модели  
  
-----------
### Самооценка  
Выполнен минимум из требуемых задач.  
  
Самооценка:
  
0. Поднимите airflow локально, используя docker compose (можно использовать из примера https://github.com/made-ml-in-prod-2021/airflow-examples/)   
Выполнено  
1. (5 баллов) Реализуйте dag, который генерирует данные для обучения модели (генерируйте данные, можете использовать как генератор синтетики из первой дз, так и что-то из датасетов sklearn), вам важно проэмулировать ситуации постоянно поступающих данных
- записывайте данные в /data/raw/{{ ds }}/data.csv, /data/raw/{{ ds }}/target.csv  
Выполнено  
2. (10 баллов) Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день. В вашем пайплайне должно быть как минимум 4 стадии, но дайте волю своей фантазии=)  
Выполнено  
- подготовить данные для обучения(например, считать из /data/raw/{{ ds }} и положить /data/processed/{{ ds }}/train_data.csv)
- расплитить их на train/val
- обучить модель на train (сохранить в /data/models/{{ ds }} 
- провалидировать модель на val (сохранить метрики к модельке)

3. Реализуйте dag, который использует модель ежедневно (5 баллов)
- принимает на вход данные из пункта 1 (data.csv)
- считывает путь до модельки из airflow variables(идея в том, что когда нам нравится другая модель и мы хотим ее на прод 
- делает предсказание и записывает их в /data/predictions/{{ds }}/predictions.csv  
Выполнено частично  
3.1  Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения (3 доп балла)  
Не выполнено  
4. вы можете выбрать 2 пути для выполнения ДЗ. 
- поставить все необходимые пакеты в образ с airflow и использовать bash operator, python operator (0 баллов)
- использовать DockerOperator, тогда выполнение каждой из тасок должно запускаться в собственном контейнере
-- 1 из дагов реализован с помощью DockerOperator (5 баллов)
-- все даги реализованы только с помощью DockerOperator (10 баллов) (пример https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py).  
Выполнено  
По технике, вы можете использовать такую же структуру как в примере, пакую в разные докеры скрипты, можете использовать общий докер с вашим пакетом, но с разными точками входа для разных тасок. 
Прикольно, если вы покажете, что для разных тасок можно использовать разный набор зависимостей. 

https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py#L27 в этом месте пробрасывается путь с хостовой машины, используйте здесь путь типа /tmp или считывайте из переменных окружения.

5. Протестируйте ваши даги (5 баллов) https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html   
Не выполнено  
6. В docker compose так же настройте поднятие mlflow и запишите туда параметры обучения, метрики и артефакт(модель) (5 доп баллов)  
Не выполнено  
7. вместо пути в airflow variables  используйте апи Mlflow Model Registry (5 доп баллов)  
Даг для инференса подхватывает последнюю продакшен модель.   
Не выполнено  
8. Настройте alert в случае падения дага (3 доп. балла)
https://www.astronomer.io/guides/error-notifications-in-airflow  
Не выполнено  
9. традиционно, самооценка (1 балл)  
Выполнено  
  
5+10+5+10+1 = 26-31  
-40% за хард дедлайн ~= 15.5-18.5  
Разброс из-за задачи 3, я не до конца разобрался как использовать variables, поэтому сделал просто отдельную папку с последней тренированной моделью.  
  