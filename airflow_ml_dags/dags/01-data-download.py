from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from dag_constants import START_DATE, DEFAULT_ARGS,\
    DATA_VOLUME_DIR, RAW_DATA_DIR

with DAG(
    "01-data_download",
    default_args=DEFAULT_ARGS,
    start_date=START_DATE,
    schedule_interval="@daily",
) as dag:

    start_download = DummyOperator(task_id='begin-download-data')

    data_download = DockerOperator(
        image="airflow-download",
        command=f"--output-dir {RAW_DATA_DIR}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="data-download",
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    start_download >> data_download
