from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .features import build_pipeline


def model_catalog(random_state=42):
    return {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        "Random Forest": build_pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1
        )),
        "XGBoost": build_pipeline(XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )),
    }


def tuned_xgb(random_state=42):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=random_state,
        n_jobs=-1, tree_method="hist",
    )
