from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def create_json_results(
    y_test: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels_map: dict,
) -> dict:
    """
    Create a JSON object with the results
    :param y_test: List of test values
    :param y_pred: List of pred values
    :param labels_map: Dict of labels
    :return: Dict of results
    """
    reverse_labels_map = {v: k for k, v in labels_map.items()}
    y_test = [reverse_labels_map[x] for x in y_test]
    y_pred = [reverse_labels_map[x] for x in y_pred]
    results = {"Test": y_test, "Pred": y_pred}
    return results


def get_scores(
    y_test: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> tuple[float, float]:
    """
    Get the accuracy and F1 scores
    :param y_test: List of test values
    :param y_pred: List of pred values
    :return: Dict of scores
    """
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    return accuracy, f1


def get_average_scores(
    f1_scores: list[float], accuracies: list[float]
) -> tuple[float, float]:
    """
    Get the average accuracy and F1 scores
    :param f1_scores: List of F1 scores
    :param accuracies: List of accuracies
    :return: Tuple of average accuracy and F1 scores
    """
    avr_f1_macro = sum(f1_scores) / len(f1_scores)
    avr_accuracy = sum(accuracies) / len(accuracies)
    return avr_accuracy, avr_f1_macro


def confusion_matrix_heatmap(
    y_test: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels_map: dict,
    title: str,
) -> plt.figure:
    """
    Plot the confusion matrix heatmap
    :param y_test: List of test values
    :param y_pred: List of pred values
    :param labels: Dict of labels
    :param title: Title of the plot
    """
    reverse_labels_map = {v: k for k, v in labels_map.items()}
    y_test = [reverse_labels_map[x] for x in y_test]
    y_pred = [reverse_labels_map[x] for x in y_pred]
    labels_values = list(set(y_pred) | set(y_test))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels_values, columns=labels_values)
    plt.figure(figsize=(10, 10))
    matrix = sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="g")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    return matrix
