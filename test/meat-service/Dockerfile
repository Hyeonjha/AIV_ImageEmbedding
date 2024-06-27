FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install confluent_kafka
CMD ["python", "main.py"]
