import os

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from dag_constants import START_DATE, DEFAULT_ARGS, \
    DATA_VOLUME_DIR, RAW_DATA_DIR, PREDICT_DIR, MODEL_PATH

with DAG(
        "03-ml_predict",
        default_args=DEFAULT_ARGS,
        start_date=START_DATE,
        default_view="graph",
        schedule_interval="@daily",
) as dag:
    start_predict = DummyOperator(task_id='begin-predict')

    wait_data = FileSensor(
        task_id="wait-for-data",
        filepath=str(os.path.join(RAW_DATA_DIR, "data.csv")),
        timeout=3600,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {RAW_DATA_DIR} "
                f"--output-dir {PREDICT_DIR} "
                f"--model-path {MODEL_PATH}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="predict",
        volumes=[f"{DATA_VOLUME_DIR}:/data"]
    )

    start_predict >> wait_data >> predict
