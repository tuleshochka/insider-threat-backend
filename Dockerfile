FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional dependencies for ML and Postgres
RUN pip install --no-cache-dir psycopg2-binary redis celery xgboost lightgbm imbalanced-learn shap

COPY . .

# FastAPI runs on port 8000
EXPOSE 8000
