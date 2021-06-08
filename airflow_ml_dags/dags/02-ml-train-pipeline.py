import os

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from dag_constants import START_DATE, DEFAULT_ARGS, \
    DATA_VOLUME_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, \
    TRAIN_SIZE, SPLIT_SEED, TRAIN_SEED

with DAG(
        "02-ml_train_pipeline",
        default_args=DEFAULT_ARGS,
        start_date=START_DATE,
        default_view="graph",
        schedule_interval="@weekly",
) as dag:
    start_preprocess = DummyOperator(task_id='begin-ml-train')

    wait_data = FileSensor(
        task_id="wait-for-data",
        filepath=str(os.path.join(RAW_DATA_DIR, "data.csv")),
        timeout=3600,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    wait_target = FileSensor(
        task_id="wait-for-target",
        filepath=str(os.path.join(RAW_DATA_DIR, "target.csv")),
        timeout=3600,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    data_preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {RAW_DATA_DIR} "
                f"--output-dir {PROCESSED_DATA_DIR}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="data-preprocess",
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
        entrypoint="python preprocess.py"
    )

    data_split = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {PROCESSED_DATA_DIR} "
                f"--train-size {TRAIN_SIZE} --seed {SPLIT_SEED}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="data-split",
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
        entrypoint="python split.py"
    )

    train_model = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {PROCESSED_DATA_DIR} "
                f"--models-dir {MODELS_DIR} --seed {TRAIN_SEED}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="train_model",
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
        entrypoint="python train.py"
    )

    validate_model = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {PROCESSED_DATA_DIR} "
                f"--models-dir {MODELS_DIR}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="validate_model",
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
        entrypoint="python validate.py"
    )

    start_preprocess >> \
        [wait_data, wait_target] >> \
        data_preprocess >> \
        data_split >> \
        train_model >> \
        validate_model
