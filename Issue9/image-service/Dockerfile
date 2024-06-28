FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
COPY wait-for-it.sh /usr/local/bin/
CMD ["wait-for-it.sh", "db:5432", "--", "flask", "run", "--host=0.0.0.0", "--port=5000"]
