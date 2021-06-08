import pytest
from airflow.models import DagBag


@pytest.fixture()
def dag_bag():
    dag = DagBag(dag_folder='dags/', include_examples=False)
    return dag


@pytest.mark.parametrize(
    "dag_id, num_tasks",
    [
        pytest.param("01-data_download", 2),
        pytest.param("02-ml_train_pipeline", 7),
        pytest.param("03-ml_predict", 3),
    ],
)
def test_dag_loaded(dag_bag, dag_id, num_tasks):
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == num_tasks
