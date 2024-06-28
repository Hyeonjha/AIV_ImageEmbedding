FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
COPY wait-for-it.sh /usr/local/bin/
CMD ["wait-for-it.sh", "kafka:9092", "--", "python", "main.py"]
