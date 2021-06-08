import os
from datetime import timedelta

from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

START_DATE = days_ago(14)

DATA_VOLUME_DIR = "/media/andyst/STORE/Notebook/ml_in_prod/airflow_ml_dags" \
                  "/data"

RAW_DATA_DIR = "/data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODELS_DIR = "/data/models/{{ ds }}"
PREDICT_DIR = "/data/predictions/{{ ds }}"
MODEL_PATH = os.environ["MODEL_PATH"]

SPLIT_SEED = 1234
TRAIN_SEED = 1234

TRAIN_SIZE = 0.8
