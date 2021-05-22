import os

from fastapi.testclient import TestClient
import pytest

from src.main_app import app, settings


DEFAULT_SUCCESS_STATUS_CODE = 200
DEFAULT_VALIDATION_ERROR_CODE = 400

GOOD_FEATURES = ["slope", "oldpeak", "sex", "trestbps", "ca", "restecg",
                 "chol", "age", "thal", "thalach", "fbs", "cp", "exang"]

GOOD_DATA_1 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
ANSWER_DATA_1 = [0, 1]

GOOD_DATA_2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]]
ANSWER_DATA_2 = [0, 0]

BAD_DATA = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

BAD_FEATURES = ["slope", "oldpeak", "sex", "trestbps", "ca", "restecg"]

test_settings = settings
test_settings.config_path = os.path.join("configs", "app_conf.yaml")


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    return TestClient(app)


def test_root(test_client):
    response = test_client.get("/")
    assert response.status_code == DEFAULT_SUCCESS_STATUS_CODE


def test_predict_bad_req(test_client):
    response = test_client.post("/predict",
                                json={"bad_features": []}
                                )
    assert response.status_code == DEFAULT_VALIDATION_ERROR_CODE


@pytest.mark.parametrize(
    "data, answer",
    [
        pytest.param(GOOD_DATA_1, ANSWER_DATA_1, id="Good dataset [0, 1]"),
        pytest.param(GOOD_DATA_2, ANSWER_DATA_2, id="Good dataset [0, 0]"),
    ],
)
def test_predict_good_req(test_client, data, answer):
    response = test_client.post("/predict",
                                json={"features": GOOD_FEATURES,
                                      "data": data}
                                )
    assert response.status_code == DEFAULT_SUCCESS_STATUS_CODE

    response_data = response.json()

    assert isinstance(response_data, dict)
    assert "predict" in response_data

    data = response_data["predict"]
    assert isinstance(data, list)
    assert len(GOOD_DATA_1) == len(data)

    assert data == answer


def test_predict_bad_data_req(test_client):
    response = test_client.post("/predict",
                                json={"features": GOOD_FEATURES,
                                      "data": BAD_DATA}
                                )
    assert response.status_code == DEFAULT_VALIDATION_ERROR_CODE


def test_predict_bad_features_req(test_client):
    response = test_client.post("/predict",
                                json={"features": BAD_FEATURES,
                                      "data": GOOD_DATA_1}
                                )
    assert response.status_code == DEFAULT_VALIDATION_ERROR_CODE
