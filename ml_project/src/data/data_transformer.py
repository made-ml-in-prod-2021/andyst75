import pandas as pd
import numpy as np

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

    def fit(self, data: pd.DataFrame):
        self.categorical.fit(data[self.categorical_features].values)
        self.numerical.fit(data[self.numerical_features].values)
        return self

    def transform(self, data: pd.DataFrame):
        num_scaler = \
            self.numerical.transform(
                data[self.numerical_features].values)
        cat_ohe = \
            self.categorical.transform(
                data[self.categorical_features].values)
        transformed_data = np.hstack([num_scaler, cat_ohe])
        return transformed_data
