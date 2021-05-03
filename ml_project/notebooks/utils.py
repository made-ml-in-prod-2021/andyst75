from typing import List

import pandas.io.formats.style as style
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

COLUMNS_NEW_DESC = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest_pain',
                    'trestbps': 'Resting_blood_pressure',
                    'chol': 'Cholesterol',
                    'fbs': 'Fasting_blood_sugar', 'restecg': 'ECG_results',
                    'thalach': 'Maximum_heart_rate',
                    'exang': 'Exercise_induced_angina',
                    'oldpeak': 'ST_depression', 'ca': 'Major_vessels',
                    'thal': 'Thalassemia_types', 'target': 'Heart_attack',
                    'slope': 'ST_slope'}

TARGET_COLUMN = 'Heart_attack'

NUM_COLUMNS = ['Age', 'Sex', 'Resting_blood_pressure', 'Cholesterol',
               'Fasting_blood_sugar',
               'Maximum_heart_rate', 'Exercise_induced_angina',
               'ST_depression']

CAT_COLUMNS = ['Chest_pain', 'Thalassemia_types', 'ECG_results', 'ST_slope',
               'Major_vessels']


def build_heatmap_analysis(df: DataFrame,
                           groupby_list: List[str],
                           column: str, cmap: str,
                           display_lines: int = 10) -> style.Styler:
    """
    Return display_lines stylish heatmap from pandas DataFrame,
    with groupby by groupby_list, target column and cmap
    """

    return df.groupby(groupby_list)[column] \
        .count() \
        .reset_index() \
        .sort_values(by=column, ascending=False) \
        .head(display_lines) \
        .style.background_gradient(cmap=cmap)


def search_best_parameters(estimator: BaseEstimator, parameters: dict,
                           folds: int,
                           x_train: DataFrame, y_train: DataFrame,
                           n_jobs: int = 8) -> dict:
    """ Find optimal parameters for estimator """

    model_cv = GridSearchCV(estimator, parameters,
                            scoring='neg_mean_squared_error', cv=folds,
                            n_jobs=n_jobs)
    model_cv.fit(x_train, y_train)
    return model_cv.best_params_


def build_model(estimator: BaseEstimator,
                x_train: DataFrame, y_train: DataFrame,
                x_test: DataFrame, y_test: DataFrame) -> \
        (BaseEstimator, float):
    """ Build model and calc accuracy"""

    model = estimator
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return model, accuracy
