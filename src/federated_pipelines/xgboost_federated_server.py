import argparse
import os
import sys
import tempfile

import flwr as fl
import mlflow as ml
from flwr.server.strategy import FedXgbBagging

sys.path.append(os.path.abspath(os.path.join("..")))

# from utils.validation import plot_metrics

AVERAGE_F1_MACROS = []
AVERAGE_ACCURACIES = []
BEST_ROUND = 1
BEST_F1_MACRO = 0
BEST_ACCURACY = 0


def evaluate_metrics_aggregation(eval_metrics: list) -> dict:
    """
    Aggregate evaluation metrics from all clients.
    :param eval_metrics: List of evaluation metrics from all clients.
    :return: Aggregated evaluation metrics.
    """
    global BEST_ROUND, BEST_F1_MACRO, BEST_ACCURACY
    total_num = sum([num for num, _ in eval_metrics])
    acc_aggregated = (
        sum([metric["accuracy"] * num for num, metric in eval_metrics]) / total_num
    )
    f1_aggregated = (
        sum([metric["f1_macro"] * num for num, metric in eval_metrics]) / total_num
    )
    metrics_aggregated = {"accuracy": acc_aggregated, "f1_macro": f1_aggregated}
    AVERAGE_F1_MACROS.append(f1_aggregated)
    AVERAGE_ACCURACIES.append(acc_aggregated)
    if f1_aggregated > BEST_F1_MACRO:
        BEST_F1_MACRO = f1_aggregated
        BEST_ACCURACY = acc_aggregated
        BEST_ROUND = len(AVERAGE_F1_MACROS)
    return metrics_aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number-of-clients", type=int, help="Number of clients.", required=True
    )
    parser.add_argument(
        "--number-of-rounds", type=int, help="Number of rounds.", required=True
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Name of the experiment.", required=True
    )
    num_clients = parser.parse_args().number_of_clients
    num_rounds_global = parser.parse_args().number_of_rounds
    experiment_name = parser.parse_args().experiment_name

    pool_size = num_clients
    num_rounds = num_rounds_global
    num_clients_per_round = num_clients
    num_evaluate_clients = num_clients

    strategy = FedXgbBagging(
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    print("Starting Flower server...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    run_name = f"xgboost_federated_{num_clients}_clients_{num_rounds}_rounds"
    average_f1 = sum(AVERAGE_F1_MACROS) / len(AVERAGE_F1_MACROS)
    average_acc = sum(AVERAGE_ACCURACIES) / len(AVERAGE_ACCURACIES)
    # avr_f1_plot = plot_metrics(
    #     AVERAGE_F1_MACROS,
    #     "F1 Macro",
    #     "F1 Macro per round",
    # )
    # avr_accuracy_plot = plot_metrics(
    #     AVERAGE_ACCURACIES,
    #     "Accuracy",
    #     "Accuracy per round",
    # )
    ml.set_experiment(experiment_name)
    with ml.start_run(run_name=run_name):
        ml.log_params(
            {
                "Number of Clients": num_clients,
                "Number of Rounds": num_rounds_global,
            }
        )
        ml.log_metrics(
            {
                "Average Accuracy": average_acc,
                "Average F1 Macro": average_f1,
                "Best Accuracy": BEST_ACCURACY,
                "Best F1 Macro": BEST_F1_MACRO,
                "Best Round": BEST_ROUND,
            }
        )
        # with tempfile.NamedTemporaryFile(suffix=".png") as file:
        #     avr_f1_plot.savefig(file.name)
        #     ml.log_artifact(file.name, "average_f1_macro.png")
        # with tempfile.NamedTemporaryFile(suffix=".png") as file:
        #     avr_accuracy_plot.savefig(file.name)
        #     ml.log_artifact(file.name, "average_accuracy.png")
