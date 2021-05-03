import pandas as pd

import hydra

from sklearn.base import BaseEstimator, TransformerMixin

from ..classes import TransformParams
from ..classes import FeatureParams


class DatasetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feature_param: FeatureParams,
                 trans_param: TransformParams):
        self.categorical_features = feature_param.categorical_features
        self.numerical_features = feature_param.numerical_features
        self.categorical = \
            hydra.utils.instantiate(trans_param.categorical_transform,
                                    **trans_param.categorical_parameters)
        self.numerical = \
            hydra.utils.instantiate(trans_param.numerical_transform,
                                    **trans_param.numerical_parameters)
        print(type(self.numerical))

    def fit(self, data: pd.DataFrame, target: pd.DataFrame = None):
        print('cat', self.categorical_features)
        print('num', self.numerical_features)
        print(data[self.categorical_features].values.shape)
        print(data[self.numerical_features].values.shape)
        print(data.shape)

        self.categorical.fit(data[self.categorical_features].values)
        self.numerical.fit(data[self.numerical_features].values)

        return self

    def transform(self, data: pd.DataFrame, target: pd.DataFrame = None):
        transformed_data = data.copy()
        return transformed_data, target
