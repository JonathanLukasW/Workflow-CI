FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow pandas numpy scikit-learn matplotlib

COPY . /app

CMD ["mlflow", "run", ".", "--env-manager=local"]