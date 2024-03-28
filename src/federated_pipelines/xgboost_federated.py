import itertools
import multiprocessing
import os
import time


def start_server(number_round: str, number_clients: str, experiment_name: str):
    """
    Starts the server
    :param number_clients: Number of clients
    :param number_round: Number of rounds
    :param experiment_name: Name of the experiment
    """
    os.system(
        f"python xgboost_federated_server.py --number-of-clients {number_clients} --number-of-rounds {number_round} --experiment-name {experiment_name}"
    )


def start_client(dataset_path: str):
    """
    Starts a client
    :param dataset_path: Path to the dataset
    """
    print(f"Starting client for dataset {dataset_path}")
    os.system(
        f"python xgboost_federated_client.py --dataset-path {dataset_path} --client-id {dataset_path.split(os.sep)[-1].split('.')[0]}"
    )


if __name__ == "__main__":
    dataset_path = r"D:\Work\Federated-Network-Intrusion-Detection\src\dataset\processed_dataset"
    switches = [
        "of_000000000000000c",
        "of_000000000000000a",
        "of_000000000000000b",
        "of_0000000000000003",
        "of_0000000000000004",
    ]
    number_rounds = [3, 5, 8, 10, 15, 20, 30]
    number_clients = [2, 3, 4, 5]
    combinations = list(itertools.product(number_rounds, number_clients))
    for number_round, number_client in combinations:
        print(
            f"Starting experiment with {number_round} rounds and {number_client} clients"
        )
        server_process = multiprocessing.Process(
            target=start_server,
            args=(number_round, number_client, "XGBoost_Federated"),
        )
        server_process.start()
        time.sleep(15)

        with multiprocessing.Pool(number_client) as pool:
            csv_files = [
                os.path.join(dataset_path, data)
                for data in os.listdir(dataset_path)
                if data.endswith(".csv")
                and "all" not in data
                and any(switch in data for switch in switches)
            ]
            pool.map(start_client, csv_files)

        server_process.join()
    print("Done")
