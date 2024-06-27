FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install flask confluent_kafka
CMD ["python", "app.py"]
