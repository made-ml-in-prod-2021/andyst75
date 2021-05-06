import os
from textwrap import dedent
from typing import Tuple

import pandas as pd
import pytest
from hydra.experimental import initialize_config_dir, compose
from omegaconf import OmegaConf

from src.classes import ConfigParams, ModelReport
from src.train import train_pipeline
from src.predict import predict
from .constats import FAKE_DATASET_FULL_SIZE, FAKE_DATASET_NAME
from .fake_data import make_data


@pytest.fixture()
def test_data_train_predict_config(tmpdir) -> Tuple[ConfigParams, str, str]:
    yaml_conf = dedent("""\
        models:
            model:
              _target_: "sklearn.ensemble.RandomForestClassifier"
            model_parameters:
              "criterion": "gini"
              "max_features": "log2"
              "n_estimators": 150
              "random_state": 111
            
            transforms:
              numerical_transform:
                _target_: "sklearn.preprocessing.MinMaxScaler"
              numerical_parameters:
                  copy: True
              categorical_transform:
                _target_: "sklearn.preprocessing.OneHotEncoder"
              categorical_parameters:
                  handle_unknown: "ignore"
                  sparse: False
        
        features:
            target_col: "target"
            categorical_features:
              - "cp"
              - "thal"
              - "restecg"
              - "slope"
              - "ca"
            numerical_features:
              - "age"
              - "sex"
              - "trestbps"
              - "chol"
              - "fbs"
              - "thalach"
              - "exang"
              - "oldpeak"
        
        split:
            val_size: 0.2
            random_state: 5
        
        input_data_path: "{}"
        model_path: "{}"
        report_path: "{}"
        log_path: "configs/log_test.yaml"
        
        transform_path:
          numerical: "{}"
          categorical: "{}"
    """)

    df_path = tmpdir.join(FAKE_DATASET_NAME)
    df = make_data(FAKE_DATASET_FULL_SIZE)
    df.to_csv(df_path)

    model_path = tmpdir.join("test_model.pkl")
    report_path = tmpdir.join("test_report.yaml")
    numerical_transform_path = tmpdir.join("numerical_transform_path.pkl")
    categorical_transform_path = tmpdir.join("categorical_transform_path.pkl")

    yaml_conf = yaml_conf.format(df_path,
                                 model_path,
                                 report_path,
                                 numerical_transform_path,
                                 categorical_transform_path)

    tmp_file = tmpdir.join("test_config.yaml")

    with open(tmp_file, "w") as file_out:
        file_out.write(yaml_conf)

    with initialize_config_dir(config_dir=tmp_file.dirname,
                               job_name="test_app"):
        cfg = compose(config_name="test_config")

    config_param = ConfigParams(**cfg)
    os.remove(tmp_file)

    yaml_conf = dedent("""\
        log_config: "configs/log_predict.yaml"

        model: "{}"

        transform_path:
          numerical: "{}"
          categorical: "{}"

        features: "configs/features/5cat_8num.yaml"

        data_path: "{}"
        predict_path: "{}"
    """)

    predict_path = os.path.join(tmpdir, "predict.csv")

    yaml_conf = yaml_conf.format(config_param.model_path,
                                 config_param.transform_path.numerical,
                                 config_param.transform_path.categorical,
                                 config_param.input_data_path,
                                 predict_path
                                 )

    predict_config_path = tmpdir.join("test_predict_config.yaml")

    with open(predict_config_path, "w") as file_out:
        file_out.write(yaml_conf)

    return config_param, predict_config_path, predict_path


def test_train_predict_pipeline(
        test_data_train_predict_config: Tuple[ConfigParams, str, str]):
    test_data_config, predict_path_config, predict_path = \
        test_data_train_predict_config

    train_pipeline(test_data_config)
    report_dict = OmegaConf.load(test_data_config.report_path)
    report = ModelReport(**report_dict)

    assert report.f1_metric > 0.5

    predict(predict_path_config)
    pred_df = pd.read_csv(predict_path)

    assert len(pred_df) == FAKE_DATASET_FULL_SIZE
