Курс "Машинное обучение в продакшене":
https://data.mail.ru/blog/view/339/

Студент:
https://data.mail.ru/profile/v.zenin/

### Домашее задание 2  
-----------

Сборка докер образа из расположения `online_inference\docker`:  
	`docker build -f Dockerfile -t made/hw1 ../..`
  
Запуск докер образа:  
	`docker run -it -p 8050:8050 made/hw1`  
  
Страница документации api по умолчанию:  
	`http://127.0.0.1:8050/docs`  
  
Реализованы post-запросы для отдельного элемента и для батча элементов.  
  
### Тестирование  
Реализовано автоматическое тестирование при запуске сервера в контейнере, при помощи pytest.
Так же выполнить тестирование методов можно прямо на странице доков, нажав try it now, оставить как есть или изменить тело запроса и нажать Execute.  
Ответ появится в поле Response body.  
В папке server лежит 4 json файла с валидными и не валидными примерами данных.  
Кроме того, можно воспользоваться расширением для хрома Talend API Tester.  
  
### Оптимизация образа  
	- requirements сформированы с помощью pipreqs, что обеспечивает использование только нужных зависимостей.  
	- оптимизирован порядок сборки (есть комментарии в докерфайле).  
	- команды RUN объединены в одну с целью сокращения количества слоев.  
	- ненужные файлы добавлены в .dockerignore однако из-за маленького веса файлов проекта, это не существенно.  
	- pip --no-cache-dir позволило сократить размер образа более чем на 100 мб.  
	- python slim позволил сократить размер образа вдвое.  
  
### Образ опубликован  
https://hub.docker.com/repository/docker/wladimir90/made  
  
### Самооценка  
Оценил на 22 балла:  
1. ветку назовите homework2, положите код в папку online_inference  
Выполнено  
2. Оберните inference вашей модели в rest сервис(вы можете использовать как FastAPI, так и flask, другие желательно не использовать, дабы не плодить излишнего разнообразия для проверяющих), должен быть endpoint /predict (3 балла)  
Выполнено  
3. Напишите тест для /predict  (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/)  
Выполнено  
4. Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла  
Выполнено совместно с п.2 - тест в отдельном скрипте, делает запросы к серверу  
5. Сделайте валидацию входных данных (например, порядок колонок не совпадает с трейном, типы не те и пр, в рамках вашей фантазии)  (вы можете сохранить вместе с моделью доп информацию, о структуре входных данных, если это нужно) -- 3 доп балла  
https://fastapi.tiangolo.com/tutorial/handling-errors/ -- возращайте 400, в случае, если валидация не пройдена  
Выполнено - реализуется автоматически при помощи pydantic  
6. Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл)  
Выполнено  
7. Оптимизируйте размер docker image (3 доп балла) (опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться)  -- https://docs.docker.com/develop/develop-images/dockerfile_best-practices/  
Выполнено.  
8. опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла)  
Выполнено. https://hub.docker.com/repository/docker/wladimir90/made  
9. напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл)
Убедитесь, что вы можете протыкать его скриптом из пункта 3  
Выполнено автоматическим способом. При запуске образа, в первую очередь прогоняется тест, его результат виден в консоли. Потом стартует сервер.  
10. проведите самооценку -- 1 доп балл  
Выполнено  
11. создайте пулл-реквест и поставьте label -- hw2  
Выполнено  
  
3+3+2+3+4+3+2+1+1 = 22 баллов
