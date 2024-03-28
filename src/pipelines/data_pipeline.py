import os
import logging
import pandas as pd

from utils.data_preparation import (
    cleanse_data,
    drop_unused_columns,
    fix_bad_column,
    map_labels,
    map_ports,
    split_data,
)

PORT_MAPPING = {"Port#:1": 1, "Port#:2": 2, "Port#:3": 3, "Port#:4": 4}
BINARY_LABELS_MAPPING = {"Normal": 0, "Attack": 1}
MULTI_CLASS_LABELS_MAPPING = {
    "Normal": 0,
    "TCP-SYN": 1,
    "Blackhole": 2,
    "Diversion": 3,
    "Overflow": 4,
    "PortScan": 5,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def process_data(data_path: str) -> pd.DataFrame:
    """
    Process the data for ML model training
    :param data_path: str
    :return: pd.DataFrame
    """
    dataframe = pd.read_csv(data_path)
    dataframe = cleanse_data(dataframe)
    dataframe = map_labels(
        dataframe, BINARY_LABELS_MAPPING, MULTI_CLASS_LABELS_MAPPING
    )
    dataframe = map_ports(dataframe, PORT_MAPPING)
    dataframe = drop_unused_columns(dataframe)
    dataframe = fix_bad_column(dataframe)
    return dataframe


if __name__ == "__main__":
    data_path = "/home/bojan-emteq/Work/Federated-Network-Intrusion-Detection/src/dataset/initial_dataset/UNR-IDD.csv"
    processed_data_path = "/home/bojan-emteq/Work/Federated-Network-Intrusion-Detection/src/dataset/processed_dataset"
    os.makedirs(processed_data_path, exist_ok=True)
    logger.info("Processing the data")
    dataframe = process_data(data_path)
    logger.info("Data Processed")
    logger.info("Splitting the data")
    split_data(dataframe, processed_data_path)
    logger.info("Data Split")
