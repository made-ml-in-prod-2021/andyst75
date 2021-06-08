import os

from airflow import DAG
from airflow.models import DagRun
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from dag_constants import START_DATE, DEFAULT_ARGS, \
    DATA_VOLUME_DIR, RAW_DATA_DIR, PREDICT_DIR, MODELS_DIR, TRAIN_DAG


def get_most_recent_dag_run(dag_id):
    dag_runs = DagRun.find(dag_id=dag_id)
    dag_runs.sort(key=lambda x: x.execution_date, reverse=True)
    return str(dag_runs[0].execution_date)[:10] if dag_runs else None


with DAG(
        "04-ml_predict_last_model",
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

    last_date = get_most_recent_dag_run(TRAIN_DAG)
    if last_date is None:
        start_predict >> wait_data
    else:
        model_path = str(os.path.join(MODELS_DIR[:-8] + last_date,
                                      "model.pkl"))
        predict = DockerOperator(
            image="airflow-predict",
            command=f"--input-dir {RAW_DATA_DIR} "
                    f"--output-dir {PREDICT_DIR} "
                    f"--model-path {model_path}",
            network_mode="bridge",
            do_xcom_push=False,
            task_id="predict",
            volumes=[f"{DATA_VOLUME_DIR}:/data"],
        )
        start_predict >> wait_data >> predict
