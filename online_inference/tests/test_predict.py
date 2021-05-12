import os

import numpy as np
from fastapi.testclient import TestClient
import pytest

import pytest
import pandas as pd

from src.main_app import app, settings

DEFAULT_SUCCESS_STATUS_CODE = 200
DEFAULT_VALIDATION_ERROR_CODE = 400

GOOD_FEATURES = ["slope", "oldpeak", "sex", "trestbps", "ca", "restecg",
                 "chol", "age", "thal", "thalach", "fbs", "cp", "exang"]

GOOD_DATA = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

ANSWER_DATA = [0, 1]

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


def test_predict_good_req(test_client):
    response = test_client.post("/predict",
                                json={"features": GOOD_FEATURES,
                                      "data": GOOD_DATA}
                                )
    assert response.status_code == DEFAULT_SUCCESS_STATUS_CODE

    response_data = response.json()

    assert isinstance(response_data, dict)
    assert "predict" in response_data

    data = response_data["predict"]
    assert isinstance(data, list)

    assert len(GOOD_DATA) == len(data)

    assert data == ANSWER_DATA
