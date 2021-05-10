from os.path import join
from os import remove

import pandas as pd
import pytest

from src.classes import FeatureParams, SplittingParams
from ..constats import FAKE_DATASET_SIZE, FAKE_DATASET_NAME, \
    FAKE_DATASET_SHAPE
from ..fake_data import make_data, CATEGORICAL_FEATURES, \
    NUMERICAL_FEATURES, TARGET_FEATURES

from src.data import read_data, check_data, split_train_val_data


@pytest.fixture()
def dataset(tmpdir) -> str:
    data = make_data(FAKE_DATASET_SIZE)
    path = join(tmpdir, FAKE_DATASET_NAME)
    data.to_csv(path, index=False)
    return path


def test_read_data(dataset):
    data = read_data(dataset)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == FAKE_DATASET_SIZE
    assert data.shape == FAKE_DATASET_SHAPE
    remove(dataset)
    with pytest.raises(FileNotFoundError):
        _ = read_data("fake_path")


def test_check_data():
    data = make_data(FAKE_DATASET_SIZE)
    param = {
        'target_col': TARGET_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
    }
    feature_params = FeatureParams(**param)
    check_result, _, _ = check_data(data, feature_params)
    assert check_result


def test_split_train_val_data():
    data = make_data(FAKE_DATASET_SIZE)
    val_size = 0.4
    random_state = 123
    splitting_params = {'val_size': val_size, 'random_state': random_state}
    splitting_data = SplittingParams(**splitting_params)
    df_train, df_val = split_train_val_data(data, splitting_data)

    len_train = len(df_train)
    len_val = len(df_val)

    assert len_train - 1\
           <= int(FAKE_DATASET_SIZE * (1 - val_size))\
           <= len_train + 1
    assert len_val - 1\
           <= int(FAKE_DATASET_SIZE * val_size)\
           <= len_val + 1
    assert len_train + len_val == len(data)

    columns_count = len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES) + 1
    assert df_train.shape[1] == columns_count
    assert df_val.shape[1] == columns_count
