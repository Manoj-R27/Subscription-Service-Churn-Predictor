from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BASE_CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CATEGORICAL_COLUMNS),
    ])


def build_pipeline(model):
    """Preprocessing is fitted inside each CV fold, preventing validation leakage."""
    from sklearn.pipeline import Pipeline
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def feature_names(preprocessor):
    return preprocessor.get_feature_names_out().tolist()
