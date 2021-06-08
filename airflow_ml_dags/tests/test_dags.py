import os
import sys

import pytest
from airflow.models import DagBag

sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    os.environ["DATA_FOLDER_PATH"] = "/tmp"
    os.environ["MODEL_PATH"] = "/tmp"
    return DagBag(dag_folder='dags/', include_examples=False)


@pytest.mark.parametrize(
    "dag_id, dag_flow",
    [
        pytest.param("01-data_download", {
            "begin-download-data": ["data-download"],
            "data-download": [],
        }),
        pytest.param("02-ml_train_pipeline", {
            "begin-ml-train": ["wait-for-data", "wait-for-target"],
            "wait-for-data": ["data-preprocess"],
            "wait-for-target": ["data-preprocess"],
            "data-preprocess": ["data-split"],
            "data-split": ["train_model"],
            "train_model": ["validate_model"],
            "validate_model": []
        }),
        pytest.param("03-ml_predict", {
            "begin-predict": ["wait-for-data"],
            "wait-for-data": ["predict"],
            "predict": [],
        }),
    ],
)
def test_structure_dag(dag_bag, dag_id, dag_flow):
    dag = dag_bag.dags[dag_id]

    for name, task in dag.task_dict.items():
        print("Name", name, "Task", task)
        assert task.downstream_task_ids == set(dag_flow[name])
