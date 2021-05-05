import pytest
from datetime import datetime
from sklearn.base import BaseEstimator

from src.classes import SplittingParams, FeatureParams, \
    ModelEstimator, TransformEstimator, TransformParams, TransformPath, \
    PredictParams, ModelParams, ModelReport


@pytest.fixture()
def splitting_dict():
    val_size = 0.4
    random_state = 123
    param = {'val_size': val_size, 'random_state': random_state}
    return param


@pytest.fixture()
def features_dict():
    target_col = "column_target"
    categorical_features = ["cat_1", "cat_2"]
    numerical_features = ["num_1", "num_2", "num_3"]

    param = {
        'target_col': target_col,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
    }
    return param


@pytest.fixture()
def base_estimator():
    estimator = BaseEstimator()
    return estimator


@pytest.fixture()
def transform_params_dict():
    num_estimator = BaseEstimator()
    cat_estimator = BaseEstimator()
    num_param = {'num_param1': 'par1', 'num_param2': 'par2'}
    cat_param = {'cat_param1': 1, 'cat_param2': 'par'}

    param = {
        'numerical_transform': num_estimator,
        'categorical_transform': cat_estimator,
        'numerical_parameters': num_param,
        'categorical_parameters': cat_param

    }
    return param


@pytest.fixture()
def transform_path_dict():
    categorical = 'path1'
    numerical = 'path2'

    param = {
        'categorical': categorical,
        'numerical': numerical
    }
    return param


@pytest.fixture()
def predict_config_dict(transform_path_dict: dict):
    log_config = "configs/log_config.yaml"
    model = "models/model.pkl"

    transform_path = TransformPath(**transform_path_dict)
    features = "configs/features/5cat_8num.yaml"

    data_path = "data_path/raw.csv"
    predict_path = "predict_path/out.csv"

    param = {'log_config': log_config,
             'model': model,
             'transform_path': transform_path,
             'features': features,
             'data_path': data_path,
             'predict_path': predict_path,
             }
    return param


@pytest.fixture()
def model_param_dict(base_estimator: BaseEstimator,
                     transform_params_dict: dict):

    model = base_estimator
    transforms = TransformParams(**transform_params_dict)
    model_parameters = {'lr': 1e-4}

    param = {'model': model,
             'transforms': transforms,
             'model_parameters': model_parameters
             }
    return param


@pytest.fixture()
def model_report_dict(model_param_dict: dict,
                      splitting_dict: dict):

    model = ModelParams(**model_param_dict)
    split = SplittingParams(**splitting_dict)
    data = "data.csv"
    rows = 200
    accuracy = 0.92
    f1_metric = 0.95

    param = {'model': model,
             'split': split,
             'data': data,
             'rows': rows,
             'accuracy': accuracy,
             'f1_metric': f1_metric,
             }
    return param


def test_splitting_params_valid(splitting_dict: dict):
    data = SplittingParams(**splitting_dict)
    assert data.val_size == splitting_dict['val_size']
    assert data.random_state == splitting_dict['random_state']


def test_splitting_params_invalid(splitting_dict: dict):
    splitting_dict.update({'fake_param': None})
    with pytest.raises(TypeError):
        _ = SplittingParams(**splitting_dict)


def test_features_params(features_dict: dict):
    data = FeatureParams(**features_dict)

    assert data.target_col == features_dict['target_col']
    assert data.categorical_features == features_dict['categorical_features']
    assert data.numerical_features == features_dict['numerical_features']


def test_transform_estimator_params(base_estimator: BaseEstimator):
    data = TransformEstimator(base_estimator)

    assert data._target_ == base_estimator


def test_estimator_params_valid(base_estimator: BaseEstimator):
    data = ModelEstimator(base_estimator)

    assert data._target_ == base_estimator


def test_estimator_params_invalid():
    with pytest.raises(TypeError):
        _ = ModelEstimator()


def test_transform_params(transform_params_dict: dict):
    data = TransformParams(**transform_params_dict)

    assert data.categorical_transform == \
           transform_params_dict['categorical_transform']

    assert data.numerical_transform == \
           transform_params_dict['numerical_transform']

    assert data.categorical_transform != \
           transform_params_dict['numerical_transform']

    assert data.numerical_parameters == \
           transform_params_dict['numerical_parameters']

    assert data.categorical_parameters == \
           transform_params_dict['categorical_parameters']

    assert data.numerical_parameters != \
           transform_params_dict['categorical_parameters']


def test_transform_path(transform_path_dict: dict):
    data = TransformPath(**transform_path_dict)

    assert data.categorical == transform_path_dict['categorical']
    assert data.numerical == transform_path_dict['numerical']
    assert data.categorical != transform_path_dict['numerical']


def test_predict_config(predict_config_dict: dict):
    data = PredictParams(**predict_config_dict)

    assert data.data_path == predict_config_dict['data_path']
    assert data.predict_path == predict_config_dict['predict_path']
    assert data.model == predict_config_dict['model']
    assert data.log_config == predict_config_dict['log_config']
    assert data.transform_path == predict_config_dict['transform_path']
    assert data.features == predict_config_dict['features']
    assert isinstance(data.transform_path, TransformPath)
    assert isinstance(data.model, str)


def test_model_param(model_param_dict: dict):
    data = ModelParams(**model_param_dict)

    assert data.model == model_param_dict['model']
    assert data.model_parameters == model_param_dict['model_parameters']
    assert data.transforms == model_param_dict['transforms']
    assert isinstance(data.model, BaseEstimator)


def test_model_report(model_report_dict: dict):
    data = ModelReport(**model_report_dict)

    assert data.model == model_report_dict['model']
    assert data.data == model_report_dict['data']
    assert data.split == model_report_dict['split']
    assert data.rows == model_report_dict['rows']
    assert data.f1_metric == model_report_dict['f1_metric']
    assert data.accuracy == model_report_dict['accuracy']

    assert "date" not in model_report_dict

    assert data.date[:10] == datetime.now().isoformat()[:10]
