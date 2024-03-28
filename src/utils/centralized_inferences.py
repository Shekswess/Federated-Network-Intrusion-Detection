import logging
import warnings
from abc import ABC, abstractmethod

import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from utils.validation import confusion_matrix_heatmap, get_scores
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CentralizedModel(ABC):
    """Abstract base class for centralized models."""

    @abstractmethod
    def train_test_split_LOOCV(
        self,
        dataframe: pd.DataFrame,
        exclude_columns: list[str],
        label_column: str,
        subject_column: str,
        subject_id: str,
    ):
        """Split the data for training and testing using Leave-One-Out Cross-Validation (LOOCV)."""
        pass

    def get_params(self):
        """Get the model's parameters."""
        return self.params

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model with the training data."""
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame):
        """Predict the labels for the test data."""
        pass

    @abstractmethod
    def evaluate(
        self,
        y_test: pd.Series,
        y_pred: pd.Series,
        title: str,
        class_labels_mapping: dict,
    ):
        """Evaluate the model's predictions."""
        pass


class XGBoostModel(CentralizedModel):
    """XGBoost model class."""

    default_params = {
        "eta": 0.1,
        "max_depth": 8,
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1,
        "tree_method": "hist",
    }

    def __init__(self, params: dict = None) -> None:
        """Initialize the XGBoost model with parameters."""
        if params is None:
            params = self.default_params
        self.params = params
        self.model = XGBClassifier(**params)

    def train_test_split_LOOCV(
        self,
        dataframe: pd.DataFrame,
        exclude_columns: list[str],
        label_column: str,
        subject_column: str,
        subject_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data for training and testing using Leave-One-Out Cross-Validation (LOOCV).
        """
        logger.info(f"Splitting the data for subject: {subject_id}")
        train_df = dataframe[dataframe[subject_column] != subject_id]
        test_df = dataframe[dataframe[subject_column] == subject_id]

        X_train = train_df.drop(columns=[label_column] + exclude_columns)
        y_train = train_df[label_column]

        X_test = test_df.drop(columns=[label_column] + exclude_columns)
        y_test = test_df[label_column]
        logger.info(f"Data split for subject: {subject_id}")

        return X_train, X_test, y_train, y_test

    def get_params(self):
        """Get the model's parameters."""
        return self.params

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the XGBoost model with the training data."""
        try:
            logger.info("Fitting the XGBoost model")
            self.model.fit(X_train, y_train)
            logger.info("Model fitted successfully")
        except Exception as error:
            logger.error(f"Failed to fit the model: {error}")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predict the labels for the test data."""
        try:
            logger.info("Making predictions with the XGBoost model")
            predictions = self.model.predict(X_test)
            logger.info("Predictions made successfully")
            return predictions
        except Exception as error:
            logger.error(f"Failed to make predictions: {error}")

    def evaluate(
        self,
        y_test: pd.Series,
        y_pred: pd.Series,
        title: str = "",
        class_labels_mapping: dict = None,
    ) -> dict:
        """Evaluate the XGBoost model's predictions."""
        try:
            logger.info("Evaluating the XGBoost model")
            accuracy, f1_macro = get_scores(y_test, y_pred)
            matrix = confusion_matrix_heatmap(
                y_test, y_pred, class_labels_mapping, title
            )
            results = {
                "f1_macro": f1_macro,
                "accuracy": accuracy,
                "confusion_matrix": matrix,
            }
            logger.info("Model evaluated successfully")
            return results
        except Exception as error:
            logger.error(f"Failed to evaluate the model: {error}")


class TabNetModel(CentralizedModel):

    default_network_params = {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "momentum": 0.02,
        "clip_value": 2,
        "lambda_sparse": 1e-3,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2, weight_decay=1e-5),
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_params": dict(mode="min", patience=5, factor=0.5, min_lr=1e-5),
        "mask_type": "entmax",
    }

    def __init__(
        self,
        network_params: dict = None,
    ) -> None:
        """Initialize the TabNet model with network and training parameters."""
        if network_params is None:
            network_params = self.default_network_params
        self.network_params = network_params
        self.model = TabNetClassifier(
            **network_params,
        )

    def train_test_split_LOOCV(
        self,
        dataframe: pd.DataFrame,
        exclude_columns: list[str],
        label_column: str,
        subject_column: str,
        subject_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data for training and testing using Leave-One-Out Cross-Validation (LOOCV).
        """
        logger.info(f"Splitting the data for subject: {subject_id}")
        train_df = dataframe[dataframe[subject_column] != subject_id]
        test_df = dataframe[dataframe[subject_column] == subject_id]

        X_train = train_df.drop(columns=[label_column] + exclude_columns)
        y_train = train_df[label_column]

        X_test = test_df.drop(columns=[label_column] + exclude_columns)
        y_test = test_df[label_column]
        logger.info(f"Data split for subject: {subject_id}")

        return X_train, X_test, y_train, y_test

    def get_params(self):
        """Get the model's parameters."""
        return self.network_params

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the TabNet model with the training data."""
        try:
            logger.info("Fitting the TabNet model")
            self.model.fit(
                X_train.values,
                y_train.values,
                eval_set=[(X_train.values, y_train.values)],
                patience=5,
                max_epochs=30,
            )
            logger.info("Model fitted successfully")
        except Exception as error:
            logger.error(f"Failed to fit the model: {error}")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predict the labels for the test data."""
        try:
            logger.info("Making predictions with the TabNet model")
            predictions = self.model.predict(X_test.values)
            logger.info("Predictions made successfully")
            return predictions
        except Exception as error:
            logger.error(f"Failed to make predictions: {error}")

    def evaluate(
        self,
        y_test: pd.Series,
        y_pred: pd.Series,
        title: str = "",
        class_labels_mapping: dict = None,
    ) -> dict:
        """Evaluate the TabNet model's predictions."""
        try:
            logger.info("Evaluating the TabNet model")
            accuracy, f1_macro = get_scores(y_test, y_pred)
            matrix = confusion_matrix_heatmap(
                y_test, y_pred, class_labels_mapping, title
            )
            results = {
                "f1_macro": f1_macro,
                "accuracy": accuracy,
                "confusion_matrix": matrix,
            }
            logger.info("Model evaluated successfully")
            return results
        except Exception as error:
            logger.error(f"Failed to evaluate the model: {error}")
