import logging
import warnings

import pandas as pd
from configs import MULTI_CLASS_LABELS_MAPPING, XGBOOST_CONFIG
from validation import (
    confusion_matrix_heatmap,
    create_json_results,
    get_average_scores,
    get_scores,
)
from xgboost import XGBClassifier as xgb

warnings.filterwarnings("ignore")

F1_SCORES = []
ACCURACIES = []
MATRIXES = []

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def LOOCV_XGBoost_MC(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Leave-One-Out Cross Validation for XGBoost Multi-Class
    :param dataframe: pd.DataFrame
    :return: Dict of results
    """
    json_results = {}
    logger.info("Starting Leave-One-Out Cross Validation for XGBoost Multi-Class")
    for switch_id in dataframe["Switch ID"].unique():
        logger.info(f"Processing Switch ID: {switch_id}")
        train = dataframe[dataframe["Switch ID"] != switch_id]
        test = dataframe[dataframe["Switch ID"] == switch_id]

        X_train = train.drop(["Switch ID", "Label", "Binary Label"], axis=1)
        y_train = train["Label"]

        X_test = test.drop(["Switch ID", "Label", "Binary Label"], axis=1)
        y_test = test["Label"]

        model = xgb(**XGBOOST_CONFIG)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        logger.info(f"Gettings scores for Switch ID: {switch_id}")
        accuracy, f1 = get_scores(y_test, y_pred)
        matrix = confusion_matrix_heatmap(
            y_test,
            y_pred,
            MULTI_CLASS_LABELS_MAPPING,
            f"Switch ID: {switch_id}",
        )
        json_results[switch_id] = create_json_results(
            y_test, y_pred, MULTI_CLASS_LABELS_MAPPING
        )
        ACCURACIES.append(accuracy)
        F1_SCORES.append(f1)
        MATRIXES.append(matrix)
        logger.info(f"Switch ID: {switch_id} processed successfully")

    logger.info("Leave-One-Out Cross Validation for XGBoost Multi-Class completed")
    best_accuracy = max(ACCURACIES)
    best_f1 = max(F1_SCORES)
    avr_accuracy, avr_f1_macro = get_average_scores(F1_SCORES, ACCURACIES)
    result = {
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "average_accuracy": avr_accuracy,
        "average_f1_macro": avr_f1_macro,
        "F1_scores": F1_SCORES,
        "accuracies": ACCURACIES,
        "confusion_matrices": MATRIXES,
        "json_results": json_results,
    }
    logger.info("Results computed successfully")
    return result
