import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

binary_labels_mapping = {"Normal": 0, "Attack": 1}
multi_class_labels_mapping = {
    "Normal": 0,
    "TCP-SYN": 1,
    "Blackhole": 2,
    "Diversion": 3,
    "Overflow": 4,
    "PortScan": 5,
}
bad_column_mapping = {" Delta Packets Tx Dropped": "Delta Packets Tx Dropped"}
port_mapping = {"Port#:1": 1, "Port#:2": 2, "Port#:3": 3, "Port#:4": 4}


def cleanse_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cleansing the data by removing the rows with missing values, duplicates
    :param dataframe: pd.DataFrame
    :return: pd.DataFrame
    """
    logger.info("Cleansing the data")
    dataframe = dataframe.dropna()
    dataframe = dataframe.drop_duplicates()
    logger.info("Data Cleansed")
    return dataframe


def map_labels(
    dataframe: pd.DataFrame,
    binary_label_mapping: dict,
    multi_class_label_mapping: dict,
) -> pd.DataFrame:
    """
    Maps the labels to binary and multi-class labels
    :param dataframe: pd.DataFrame
    :param binary_label_mapping: dict
    :param multi_class_label_mapping: dict
    :return: pd.DataFrame
    """
    logger.info("Mapping the labels")
    dataframe["Binary Label"] = dataframe["Binary Label"].map(binary_label_mapping)
    dataframe["Label"] = dataframe["Label"].map(multi_class_label_mapping)
    logger.info("Labels Mapped")
    return dataframe


def map_ports(dataframe: pd.DataFrame, port_mapping: dict) -> pd.DataFrame:
    """
    Maps the port numbers to integers
    :param dataframe: pd.DataFrame
    :param port_mapping: dict
    :return: pd.DataFrame
    """
    logger.info("Mapping the port numbers")
    dataframe["Port Number"] = dataframe["Port Number"].map(port_mapping)
    logger.info("Port Numbers Mapped")
    return dataframe


def drop_unused_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the columns that are not required
    :param dataframe: pd.DataFrame
    :return: pd.DataFrame
    """
    logger.info("Dropping the unused columns")
    dataframe = dataframe.drop(columns=["is_valid"])
    logger.info("Unused columns dropped")
    return dataframe


def fix_bad_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes the bad column names
    :param dataframe: pd.DataFrame
    :return: pd.DataFrame
    """
    logger.info("Fixing the bad column names")
    dataframe = dataframe.rename(columns=bad_column_mapping)
    logger.info("Bad column names fixed")
    return dataframe


def split_data(dataframe: pd.DataFrame, process_data_path: str) -> None:
    """
    Splits the data based on the SwitchID
    :param dataframe: pd.DataFrame
    :param process_data_path: str
    :return: None
    """
    logger.info("Splitting the data based on Switch ID")
    dataframe["Switch ID"] = dataframe["Switch ID"].str.replace(":", "_")
    switch_ids = dataframe["Switch ID"].unique().tolist()
    for switch_id in switch_ids:
        logger.info(f"Saving data for Switch ID: {switch_id}")
        switch_dataframe = dataframe[dataframe["Switch ID"] == switch_id]
        switch_dataframe = switch_dataframe.drop(columns=["Switch ID"])
        switch_dataframe.to_csv(
            os.path.join(process_data_path, f"{switch_id}.csv"), index=False
        )
        logger.info(f"Data saved for Switch ID: {switch_id}")
    logger.info("Saving all data to a single file")
    dataframe.to_csv(os.path.join(process_data_path, "all.csv"), index=False)
    logger.info("All data saved to a single file")


if __name__ == "__main__":
    initial_data_path = r"D:\Work\Federated-Network-Intrusion-Detection\src\initial_dataset\UNR-IDD.csv"
    process_data_path = (
        r"D:\Work\Federated-Network-Intrusion-Detection\src\processed_dataset"
    )
    os.makedirs(process_data_path, exist_ok=True)
    dataframe = pd.read_csv(initial_data_path)
    dataframe = fix_bad_column(dataframe)
    dataframe = drop_unused_columns(dataframe)
    dataframe = cleanse_data(dataframe)
    dataframe = map_labels(
        dataframe, binary_labels_mapping, multi_class_labels_mapping
    )
    dataframe = map_ports(dataframe, port_mapping)
    split_data(dataframe, process_data_path)
