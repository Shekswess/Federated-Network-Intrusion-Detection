BINARY_LABELS_MAPPING = {"Normal": 0, "Attack": 1}
MULTI_CLASS_LABELS_MAPPING = {
    "Normal": 0,
    "TCP-SYN": 1,
    "Blackhole": 2,
    "Diversion": 3,
    "Overflow": 4,
    "PortScan": 5,
}
BAD_COLUMN_MAPPING = {" Delta Packets Tx Dropped": "Delta Packets Tx Dropped"}
PORT_MAPPING = {"Port#:1": 1, "Port#:2": 2, "Port#:3": 3, "Port#:4": 4}
XGBOOST_CONFIG = {
    "eta": 0.1,
    "max_depth": 8,
    "nthreads": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}
