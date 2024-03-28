import logging
import os
import sys
import tempfile
import warnings

import mlflow as ml
import pandas as pd

sys.path.append(os.path.abspath(os.path.join("..")))

from utils.centralized_inferences import TabNetModel, XGBoostModel
from utils.validation import get_average_scores

SUBJECT_COLUMN = "Switch ID"
LABEL_COLUMN = "Label"
EXCLUDE_COLUMNS = ["Switch ID", "Binary Label", "Label"]
MULTI_CLASS_LABELS_MAPPING = {
    "Normal": 0,
    "TCP-SYN": 1,
    "Blackhole": 2,
    "Diversion": 3,
    "Overflow": 4,
    "PortScan": 5,
}
ALL_LABELS_SUBJECTS = [
    "of_000000000000000c",
    "of_000000000000000a",
    "of_000000000000000b",
    "of_0000000000000003",
    "of_0000000000000004",
]

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def XGBoost_LOOCV(dataframe: pd.DataFrame, include_subjects=None):
    """
    Train and evaluate an XGBoost model using Leave-One-Out Cross-Validation (LOOCV).
    :param dataframe: pd.DataFrame
    :return: None
    """
    f1_macro_scores = []
    accuracy_scores = []
    matrixes = []
    list_of_subject = [
        subject for subject in dataframe[SUBJECT_COLUMN].unique().tolist()
    ]
    if include_subjects:
        list_of_subject = [
            subject for subject in list_of_subject if subject in include_subjects
        ]
    for subject in list_of_subject:
        with ml.start_run(run_name=f"XGBoost_LOOCV_{subject}"):
            logger.info(f"Training and evaluating the model for {subject}")
            model = XGBoostModel()
            params = model.get_params()
            X_train, X_test, y_train, y_test = model.train_test_split_LOOCV(
                dataframe, EXCLUDE_COLUMNS, LABEL_COLUMN, SUBJECT_COLUMN, subject
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results = model.evaluate(
                y_test,
                y_pred,
                subject,
                MULTI_CLASS_LABELS_MAPPING,
            )
            ml.log_params(params)
            ml.log_metrics(
                {"f1_macro": results["f1_macro"], "accuracy": results["accuracy"]}
            )
            logger.info(f"Model trained and evaluated for {subject}")
            f1_macro_scores.append(round(results["f1_macro"], 2))
            accuracy_scores.append(round(results["accuracy"], 2))
            matrixes.append((results["confusion_matrix"], subject))
    avr_accuracy, avr_f1_macro = get_average_scores(f1_macro_scores, accuracy_scores)
    logger.info("Average scores calculated")
    with ml.start_run(run_name="XGBoost_LOOCV_Average"):
        logger.info("Logging the average scores")
        ml.log_metrics(
            {
                "f1_macro_avg": round(avr_f1_macro, 2),
                "accuracy_avg": round(avr_accuracy, 2),
            }
        )
        # for matrix, subject in matrixes:
        #     with tempfile.NamedTemporaryFile(suffix=".png") as file:
        #         matrix.get_figure().savefig(file.name)
        #         ml.log_artifact(file.name, f"confusion_matrix_{subject}.png")
        logger.info("Average scores logged")


def TabNet_LOOCV(dataframe: pd.DataFrame, include_subjects=None):
    """
    Train and evaluate a TabNet model using Leave-One-Out Cross-Validation (LOOCV).
    :param dataframe: pd.DataFrame
    :return: None
    """
    f1_macro_scores = []
    accuracy_scores = []
    matrixes = []
    list_of_subject = [
        subject for subject in dataframe[SUBJECT_COLUMN].unique().tolist()
    ]
    if include_subjects:
        list_of_subject = [
            subject for subject in list_of_subject if subject not in include_subjects
        ]
    for subject in list_of_subject:
        with ml.start_run(run_name=f"TabNet_LOOCV_{subject}"):
            logger.info(f"Training and evaluating the model for {subject}")
            model = TabNetModel()
            params = model.get_params()
            X_train, X_test, y_train, y_test = model.train_test_split_LOOCV(
                dataframe, EXCLUDE_COLUMNS, LABEL_COLUMN, SUBJECT_COLUMN, subject
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results = model.evaluate(
                y_test,
                y_pred,
                subject,
                MULTI_CLASS_LABELS_MAPPING,
            )
            ml.log_params(params)
            ml.log_metrics(
                {"f1_macro": results["f1_macro"], "accuracy": results["accuracy"]}
            )
            logger.info(f"Model trained and evaluated for {subject}")
            f1_macro_scores.append(round(results["f1_macro"], 2))
            accuracy_scores.append(round(results["accuracy"], 2))
            matrixes.append((results["confusion_matrix"], subject))
    avr_accuracy, avr_f1_macro = get_average_scores(f1_macro_scores, accuracy_scores)
    logger.info("Average scores calculated")
    with ml.start_run(run_name="TabNet_LOOCV_Average"):
        logger.info("Logging the average scores")
        ml.log_metrics(
            {
                "f1_macro_avg": round(avr_f1_macro, 2),
                "accuracy_avg": round(avr_accuracy, 2),
            }
        )
        # for matrix, subject in matrixes:
        #     with tempfile.NamedTemporaryFile(suffix=".png") as file:
        #         matrix.get_figure().savefig(file.name)
        #         ml.log_artifact(file.name, f"confusion_matrix_{subject}.png")
        logger.info("Average scores logged")


if __name__ == "__main__":
    dataframe = pd.read_csv(
        r"D:\Work\Federated-Network-Intrusion-Detection\src\dataset\processed_dataset\all.csv"
    )
    logger.info("Starting the centralized inference pipeline")
    # ml.set_experiment("XGBoost_LOOCV")
    # logger.info("Starting the XGBoost LOOCV")
    # XGBoost_LOOCV(dataframe)
    # logger.info("Ending the XGBoost LOOCV")
    # ml.set_experiment("TabNet_LOOCV")
    # logger.info("Starting the TabNet LOOCV")
    # TabNet_LOOCV(dataframe)
    # logger.info("Ending the TabNet LOOCV")
    ml.set_experiment("XGBoost LOOCV excluded subjects")
    logger.info("Starting XGBoost LOOCV with excluded subjects")
    XGBoost_LOOCV(dataframe, ALL_LABELS_SUBJECTS)
    logger.info("Ending the XGBoost LOOCV with excluded subjects")
    ml.set_experiment("TabNet LOOCV excluded subjects")
    logger.info("Starting the TabNet LOOCV with excluded subjects")
    TabNet_LOOCV(dataframe, ALL_LABELS_SUBJECTS)
    logger.info("Ending the TabNet LOOCV with excluded subjects")
    logger.info("Centralized inference pipeline completed")
