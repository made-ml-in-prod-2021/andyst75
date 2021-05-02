from typing import List
import pandas as pd
import pandas.io.formats.style as style

COLUMNS_NEW_DESC = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest_pain', 
                    'trestbps': 'Resting_blood_pressure', 'chol': 'Cholesterol',
                    'fbs': 'Fasting_blood_sugar', 'restecg': 'ECG_results', 
                    'thalach': 'Maximum_heart_rate', 'exang': 'Exercise_induced_angina',
                    'oldpeak': 'ST_depression', 'ca': 'Major_vessels', 
                    'thal': 'Thalassemia_types', 'target': 'Heart_attack',
                    'slope': 'ST_slope'}

TARGET_COLUMN = 'Heart_attack'

NUM_COLUMNS = ['Age', 'Sex', 'Resting_blood_pressure', 'Cholesterol', 'Fasting_blood_sugar',
               'Maximum_heart_rate', 'Exercise_induced_angina', 'ST_depression']
CAT_COLUMNS = ['Chest_pain', 'Thalassemia_types', 'ECG_results', 'ST_slope', 'Major_vessels']

# DROPPED_COLUMNS = ['Sex', 'Fasting_blood_sugar', 'Heart_attack', 'Chest_pain', 'ECG_results',
#                    'Exercise_induced_angina', 'ST_slope', 'ST_depression', 'Major_vessels',
#                    'Thalassemia_types']


def build_heatmap_analysis(df: pd.DataFrame, 
                           groupby_list: List[str],
                           column: str, cmap: str,
                           display_lines:int = 10) -> style.Styler:
    '''
    Return display_lines stylish heatmap from pandas DataFrame,
    with groupby by groupby_list, target column and cmap
    '''
    
    return df.groupby(groupby_list)[column] \
        .count() \
        .reset_index() \
        .sort_values(by = column, ascending = False) \
        .head(display_lines) \
        .style.background_gradient(cmap = cmap)