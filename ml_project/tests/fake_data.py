import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["cp", "thal", "restecg", "slope", "ca"]
NUMERICAL_FEATURES = ["age", "sex", "trestbps", "chol", "fbs",
                      "thalach", "exang", "oldpeak"]
TARGET_FEATURES = "target"


def build_sex(size: int) -> np.ndarray:
    a = [1, 0]
    p = [0.683168, 0.316832]
    data = np.random.choice(a, size=size, replace=True, p=p)
    return data


def build_cp(size: int) -> np.ndarray:
    a = [0, 2, 1, 3]
    p = [0.471947, 0.287129, 0.165017, 0.075907]
    data = np.random.choice(a, size=size, replace=True, p=p)
    return data


def build_thal(size: int) -> np.ndarray:
    a = [2, 3, 1, 0]
    p = [0.547855, 0.386139, 0.059406, 0.006600]
    data = np.random.choice(a, size=size, replace=True, p=p)
    return data


def build_restecg(size: int) -> np.ndarray:
    a = [1, 0, 2]
    p = [0.501650, 0.485149, 0.013201]
    data = np.random.choice(a, size=size, replace=True, p=p)
    return data


def build_slope(size: int) -> np.ndarray:
    a = [2, 1, 0]
    p = [0.468647, 0.462046, 0.069307]
    data = np.random.choice(a, size=size, replace=True, p=p)
    return data


def build_ca(size: int) -> np.ndarray:
    a = [0, 1, 2, 3, 4]
    p = [0.577558, 0.214521, 0.125413, 0.066007, 0.016501]
    data = np.random.choice(a, size=size, replace=True, p=p)
    return data


def build_target(size: int) -> np.ndarray:
    a = [0, 1]
    p = [0.455446, 0.544554]
    data = np.random.choice(a, size=size, replace=True, p=p)
    return data


def make_data(size: int, seed: int = 11) -> pd.DataFrame:
    np.random.seed(seed)
    data = pd.DataFrame()
    data['age'] = np.random.normal(loc=54.366337, scale=9.082101, size=size)
    data['sex'] = build_sex(size)
    data['cp'] = build_cp(size)
    data['trestbps'] = np.random.normal(loc=131.623762,
                                        scale=17.538143,
                                        size=size)
    data['chol'] = np.random.normal(loc=246.264026,
                                    scale=51.830751,
                                    size=size)
    data['fbs'] = np.random.normal(loc=0.148515, scale=0.356198, size=size)
    data['restecg'] = build_restecg(size)
    data['thalach'] = np.random.normal(loc=149.646865,
                                       scale=22.905161,
                                       size=size)
    data['exang'] = np.random.normal(loc=0.326733, scale=0.469794, size=size)
    data['oldpeak'] = np.random.normal(loc=1.039604, scale=1.161075, size=size)
    data['slope'] = build_slope(size)
    data['ca'] = build_ca(size)
    data['thal'] = build_thal(size)
    data['target'] = build_target(size)

    return data
