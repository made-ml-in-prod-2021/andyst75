import os
from textwrap import dedent

import pytest
from hydra.experimental import compose, initialize_config_dir

from src.classes import FeatureParams, TransformPath, \
    TransformParams
from src.data import DatasetTransformer
from ..constats import FAKE_DATASET_SIZE, FAKE_DATASET_TRANSFORM_COLUMNS
from ..fake_data import make_data, CATEGORICAL_FEATURES, \
    NUMERICAL_FEATURES, TARGET_FEATURES


@pytest.fixture()
def transform_params(tmpdir) -> TransformParams:
    yaml_conf = dedent("""\
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

    tmp_file = tmpdir.join("test_dataset_transform.yaml")

    with open(tmp_file, "w") as file_out:
        file_out.write(yaml_conf)

    with initialize_config_dir(config_dir=tmp_file.dirname,
                               job_name="test_app"):
        cfg = compose(config_name="test_dataset_transform")

    transform_param = TransformParams(**cfg)
    os.remove(tmp_file)

    return transform_param


@pytest.fixture()
def transform_path(tmpdir) -> TransformPath:
    yaml_conf = f"numerical: {tmpdir.join('test_numerical.pkl').strpath}\n" + \
                f"categorical: {tmpdir.join('test_categorical.pkl').strpath}\n"

    tmp_file = tmpdir.join("test_transform_path.yaml")

    with open(tmp_file, "w") as file_out:
        file_out.write(yaml_conf)

    with initialize_config_dir(config_dir=tmp_file.dirname,
                               job_name="test_app"):
        cfg = compose(config_name="test_transform_path")

    transform_path = TransformPath(**cfg)
    os.remove(tmp_file)

    return transform_path


def test_create_dataset_transformer_fail():
    param = {
        'target_col': TARGET_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
    }
    feature_params = FeatureParams(**param)
    with pytest.raises(NotImplementedError):
        DatasetTransformer(feature_params)


def test_create_dataset_transformer(transform_params: TransformParams,
                                    transform_path: TransformPath):
    feature_params = FeatureParams(target_col=TARGET_FEATURES,
                                   categorical_features=CATEGORICAL_FEATURES,
                                   numerical_features=NUMERICAL_FEATURES
                                   )
    trans = DatasetTransformer(feature_param=feature_params,
                               trans_param=transform_params)

    data = make_data(FAKE_DATASET_SIZE, seed=5)
    trans.fit(data)

    x_data = trans.transform(data)

    assert x_data.shape == (FAKE_DATASET_SIZE, FAKE_DATASET_TRANSFORM_COLUMNS)

    trans.dump(transform_path)

    trans = DatasetTransformer(feature_param=feature_params,
                               transform_path=transform_path)
    x_data = trans.transform(data)
    assert x_data.shape == (FAKE_DATASET_SIZE, FAKE_DATASET_TRANSFORM_COLUMNS)
