Курс "Машинное обучение в продакшене":
https://data.mail.ru/blog/view/339/

Студент:
https://data.mail.ru/profile/v.zenin/

### Домашее задание 1  
-----------

### Прототип  
Прототип решения без логирования и конфигурирования доступен по пути:
	`ml_project\notebooks\prototype.ipynb`  

### Анализ данных, построение отчета (EDA)  
Для построения отчета необходимо запустить скрипт:
	`ml_project\scripts\create_report.py`  
Отчет доступен по пути:  
	`ml_project\reports\EDA.html`  

### Тренировка модели  
А. Для тренировки модели необходимо выполнить команду (из папки `ml_project\scripts`):  
	`python train.py --config ..\configs\config.yaml`  
В результате будет сохранена тренированная модель, а по этому пути доступны метрики:  
	`ml_project\models\metrics.json`  
  
Б. Имеется возможность выбрать другой конфиг, с произвольными путями к данным, выходным файлам,
а так же настройкам разделения данных на трейн/тест:  
	`python train.py --config ..\configs\config_alt.yaml`  
  
### Получение прогноза  
А. Для получения прогноза по исходному тренировочному файлу необходимо выполнить команду (из папки `ml_project\scripts`):  
	`predict.py --config ..\configs\config.yaml`  
Результат будет доступен по пути:  
	`ml_project\models\predict.csv`  
  
Б. Для получения прогноза по произвольному файлу, требуется указать нужный файл конфига, 
в котором будет указан путь к нужному файлу, например так (из папки `ml_project\scripts`):  
	`predict.py --config ..\configs\config_alt.yaml`  
  
### Тестирование  
Для проведения тестирования необходимо перейти к папке `ml_project\tests` и выполнить следующий код:
	`pytest -m maintest`  
При этом будет выполнено 4 теста находящиеся в файлах `end_2_end_test.py` и `module_test.py`  
    
### Самооценка  
Посчитал 29 баллов.  
  
1. Назовите ветку homework1 (1 балл) - Выполнено.  
2. Положите код в папку ml_project - Выполнено.  
3. В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код. (2 балла) - Выполнено.  
4. Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 баллов) - Выполнено.  
Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)
Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл) - Выполнено.  
5. Проект имеет модульную структуру(не все в одном файле =) ) (2 баллов) - Выполнено.  
6. Использованы логгеры (2 балла) - Выполнено.  
7. Написаны тесты на отдельные модули и на прогон всего пайплайна(3 баллов) - Выполнено.  
8. Для тестов генерируются синтетические данные, приближенные к реальным (3 баллов) - Выполнено.  
9. Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла) - Выполнено.  
10. Используются датаклассы для сущностей из конфига, а не голые dict (3 балла) - Выполнено.  
11. Используйте кастомный трансформер(написанный своими руками) и протестируйте его(3 балла) - Не выполнено.  
12. Обучите модель, запишите в readme как это предлагается (3 балла) - Выполнено.  
13. напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку(без меток) и запишет предикт, напишите в readme как это сделать (3 балла)  - Выполнено.  
14. Используется hydra  (https://hydra.cc/docs/intro/) (3 балла - доп баллы) - Не выполнено.  
15. Настроен CI(прогон тестов, линтера) на основе github actions  (3 балла - доп баллы (будем проходить дальше в курсе, но если есть желание поразбираться - welcome) - Не выполнено.  
16. Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему (1 балл доп баллы) - Выполнено.  

