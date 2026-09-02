from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def save_global_shap(model, transformed, feature_names, output_path):
    values = shap.TreeExplainer(model)(transformed)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values.values, transformed, feature_names=feature_names, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def local_explanation(model, transformed_row, feature_names, max_display=10):
    values = shap.TreeExplainer(model)(transformed_row)
    ranking = sorted(zip(feature_names, values.values[0]), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "shap_value": float(value)} for name, value in ranking[:max_display]]
