from typing import List
import pandas as pd
import pandas.io.formats.style as style

def build_heatmap_analysis(df: pd.DataFrame, groupby_list: List[str], column: str, cmap: str, display_lines:int = 10) -> style.Styler:
    ''' Return display_lines stylish heatmap from pandas DataFrame, with groupby by groupby_list, target column and cmap '''
    
    return df.groupby(groupby_list)[column] \
        .count() \
        .reset_index() \
        .sort_values(by = column, ascending = False) \
        .head(display_lines) \
        .style.background_gradient(cmap = cmap)