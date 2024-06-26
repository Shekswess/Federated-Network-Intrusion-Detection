from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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


def plot_metrics(metrics: list[float], metric_name: str, title: str) -> plt.figure:
    """
    Plot the metrics
    :param metrics: List of metrics
    :param metric_name: Name of the metric
    :param title: Title of the plot
    :param save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 10))
    rounds = range(1, len(metrics) + 1)
    plt.plot(rounds, metrics)
    plt.title(title)
    plt.ylabel(metric_name)
    plt.xlabel("Round")
    plt.xticks(rounds)
    return plt
