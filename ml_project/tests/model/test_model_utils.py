import os
from textwrap import dedent
from typing import Tuple, NoReturn

import numpy as np
from hydra.experimental import initialize_config_dir, compose
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from src.classes import ModelParams
from src.model import dump_model, load_model, build_model


def make_fake_data(size: int = 5_000,
                   train_columns: int = 5,
                   seed: int = 111) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    data = np.random.uniform(0., 5., (size, train_columns))
    target = np.random.binomial(1, 0.6, size=size)

    return data, target


def test_dump_load_model(tmpdir) -> NoReturn:
    model_path = tmpdir.join("test_model.pkl").strpath
    model = RandomForestClassifier()
    wrong_model = AdaBoostClassifier()

    x_train, y_train = make_fake_data()
    x_test, _ = make_fake_data(size=200)

    model.fit(x_train, y_train)

    dump_model(model_path, model)
    loaded_model = load_model(model_path)

    wrong_model.fit(x_train, y_train)

    predict_original = model.predict(x_test)
    predict_loaded = loaded_model.predict(x_test)
    predict_wrong = wrong_model.predict(x_test)

    assert np.allclose(predict_original, predict_loaded)
    assert not np.allclose(predict_original, predict_wrong)


def test_build_model(tmpdir) -> NoReturn:
    yaml_conf = dedent("""\
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
    """)

    tmp_file = tmpdir.join("test_build_model.yaml")

    with open(tmp_file, "w") as file_out:
        file_out.write(yaml_conf)

    with initialize_config_dir(config_dir=tmp_file.dirname,
                               job_name="test_app"):
        cfg = compose(config_name="test_build_model")

    params = ModelParams(**cfg)
    os.remove(tmp_file)

    model = build_model(params)

    assert isinstance(model, RandomForestClassifier)
