FROM python:3.10.5
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python flask_covid_app.py