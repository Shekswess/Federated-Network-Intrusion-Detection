import argparse
import os
import sys
import warnings
from logging import INFO
from typing import Union

import flwr as fl
import numpy as np
import pandas as pd
import xgboost as xgb
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.logger import log
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join("..")))

from utils.validation import get_scores


class XGBoostClient(fl.client.Client):
    def __init__(
        self,
        train_dmatrix,
        test_dmatrix,
        params,
        num_local_round,
        num_train,
        num_test,
    ):
        """
        XGBoost client.
        :param train_dmatrix: The training dataset in DMatrix format.
        :param test_dmatrix: The test dataset in DMatrix format.
        :param params: The parameters for training.
        :param num_local_round: The number of local training rounds.
        :param num_train: The number of training examples.
        :param num_test: The number of test examples.
        """
        self.bst = None
        self.config = None
        self.train_dmatrix = train_dmatrix
        self.test_dmatrix = test_dmatrix
        self.params = params
        self.num_local_round = num_local_round
        self.num_train = num_train
        self.num_test = num_test

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return the local model parameters.
        :param ins: GetParametersIns
        :return: GetParametersRes
        """
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        """
        Local boosting on the local dataset.
        :return: The local model.
        """
        for i in range(self.num_local_round):
            self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - self.num_local_round : self.bst.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        """
        Train the model on the local dataset.
        :param ins: Training instructions.
        :return: Training results.
        """
        if not self.bst:
            log(INFO, "Start training at round 1")
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.test_dmatrix, "test"), (self.train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self._local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the model on the test set.
        :param ins: Evaluation instructions.
        :return: Evaluation results.
        """
        y_pred = self.bst.predict(self.test_dmatrix)
        if self.params.get("objective") == "multi:softmax":
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = [round(value) for value in y_pred]
        y_test = self.test_dmatrix.get_label()
        accuracy, f1 = get_scores(y_test, y_pred)
        results = {
            "Client ID": client_id,
            "Local Round": self.num_local_round,
            "Accuracy": accuracy,
            "F1": f1,
        }
        print(results)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_test,
            metrics={"accuracy": accuracy, "f1_macro": f1},
        )


def train_test_split_data(
    data: pd.DataFrame, test_fraction: float, seed: int
) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test set by given split rate
    and each label to be balanced.
    :param data: the dataset to be split
    :param test_fraction: the split rate
    :param seed: random seed
    :return: the train and test set
    """
    labels = data["Label"].unique()
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for label in labels:
        data_label = data[data["Label"] == label]
        data_label_train, data_label_test = train_test_split(
            data_label, test_size=test_fraction, random_state=seed
        )
        train_data = pd.concat([train_data, data_label_train])
        test_data = pd.concat([test_data, data_label_test])

    return train_data, test_data


def transform_dataset_to_dmatrix(data: pd.DataFrame) -> xgb.core.DMatrix:
    """
    Transform the dataset into DMatrix for xgboost.
    :param data: the dataset to be transformed
    :return: the transformed dataset in DMatrix format
    """
    x = data.drop(["Label", "Binary Label"], axis=1)
    y = data["Label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=str, help="Path to the dataset.", required=True
    )
    parser.add_argument("--client-id", type=str, help="Client ID.", required=True)
    args = parser.parse_args()

    client_id = args.client_id
    dataset_path = args.dataset_path
    df = pd.read_csv(dataset_path)

    train_data, test_data = train_test_split_data(df, test_fraction=0.2, seed=42)

    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    test_dmatrix = transform_dataset_to_dmatrix(test_data)

    num_local_round = 3
    params = {
        "eta": 0.1,
        "max_depth": 8,
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1,
        "tree_method": "hist",
    }

    num_train = train_dmatrix.num_row()
    num_test = test_dmatrix.num_row()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=XGBoostClient(
            train_dmatrix, test_dmatrix, params, num_local_round, num_train, num_test
        ),
    )
