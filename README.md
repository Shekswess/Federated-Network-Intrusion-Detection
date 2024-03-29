# Network-Intrusion-Detection

This is a code repository that includes all the code for the project Federated Network Intrusion Detection. The goal of this project is to compare XGBoost tabular classification model both in centralized and federated learning settings. The dataset used for this project is the UNR-IDD Intrusion Detection Dataset 2023. The dataset is used for binary and multi-class classification. 

The dataset allows us to develop two kinds of models:
- Binary Classification: The goal is to classify the network traffic as normal or attack.
- Multi-class Classification: The goal is to classify the network traffic as normal or one of the 6 attack types.

## Dataset

### Features

The dataset has 30 features. The features are divided into 3 categories:
- Port Statistics (9 features)
- Delta Port Statistics (9 features)
- Flow Entry and Flow Table Statistics (12 features)


| Port Statistic        | Description                                      |
| -------------------- | ------------------------------------------------ |
| Received Packets     | Number of packets received by the port           |
| Received Bytes       | Number of bytes received by the port             |
| Sent Packets         | Number of packets sent by the port               |
| Sent Bytes           | Number of bytes sent                             |
| Port alive Duration  | The time port has been alive in seconds          |
| Packets Rx Dropped   | Number of packets dropped by the receiver        |
| Packets Tx Dropped   | Number of packets dropped by the sender          |
| Packets Rx Errors    | Number of transmit errors                        |
| Packets Tx Errors    | Number of receive errors                         |



| Delta Port Statistic        | Description                                      |
| -------------------------- | ------------------------------------------------ |
| Delta Received Packets     | Number of packets received by the port           |
| Delta Received Bytes       | Number of bytes received by the port             |
| Delta Sent Packets         | Number of packets sent by the port               |
| Delta Sent Bytes           | Number of bytes sent                             |
| Delta Port alive Duration  | The time port has been alive in seconds          |
| Delta Packets Rx Dropped   | Number of packets dropped by the receiver        |
| Delta Packets Tx Dropped   | Number of packets dropped by the sender          |
| Delta Packets Rx Errors    | Number of transmit errors                        |
| Delta Packets Tx Errors    | Number of receive errors                         |



| Statistic               | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| Connection Point        | Network connection point expressed as a pair of the network element identifier and port number. |
| Total Load/Rate         | Obtain the current observed total load/rate (in bytes/s) on a link. |
| Total Load/Latest       | Obtain the latest total load bytes counter viewed on that link. |
| Unknown Load/Rate       | Obtain the current observed unknown-sized load/rate (in bytes/s) on a link. |
| Unknown Load/Latest     | Obtain the latest unknown-sized load bytes counter viewed on that link. |
| Time seen               | When the above-mentioned values were last seen. |
| is_valid                | Indicates whether this load was built on valid values. |
| TableID                 | Returns the Table ID values. |
| ActiveFlowEntries       | Returns the number of active flow entries in this table. |
| PacketsLookedUp         | Returns the number of packets looked up in the table. |
| PacketsMatched          | Returns the number of packets that successfully matched in the table. |
| MaxSize                 | Returns the maximum size of this table. |


### Labels

The dataset has 2 kinds of labels:
- For Binary Classification(Normal/Attack)
- For Multi-class Classification(Normal/TCP-SYN Flood/Port Scan/Flow Table Overflow/Blackhole/Traffic Diversion)

Attacks:

- TCP-SYN Flood: A Distributed Denial of Service (DDoS) attack where attackers target hosts by initiating many Transmission Control Protocol (TCP) handshake processes without waiting for the response from the target node. By doing so, the target device's resources are consumed as it has to keep allocating some memory space for every new TCP request.

- Port scan: An attack in which attackers scan available ports on a host device to learn information about the services, versions, and even security mechanisms that are running on that host. 

- Flow Table Overflow: An attack that targets network switches/routers where attacks compromise the functionality of a switch/router by consuming the flow tables that forwards packets with illegitimate flow entries and rules so that legitimate flow entries and rules cannot be installed. 

- Blackhole: An attack that targets network switches/routers to discard the packets that pass through, instead of relaying them on to the next hop. 

- Traffic Diversion: A attack that targets network switches/routers to reroute the direction of packets away from their destination, intending to increase travel time and/or spying on network traffic through a man-in-the-middle scenario.\

These intrusion types were selected for this dataset as they are common cyber attacks that can occur in any networking environment. Also, these intrusion types cover attacks that can be launched on both network devices and end hosts. 

Description of the labels:

| Label   | Description                           |
| ------- | ------------------------------------- |
| Normal  | Normal Network Functionality.         |
| Attack  | Network Intrusion.                    |


| Label      | Description                           |
| ---------- | ------------------------------------- |
| Normal     | Normal Network Functionality.         |
| TCP-SYN    | TCP-SYN Flood.                        |
| PortScan   | Port Scanning.                        |
| Overflow   | Flow Table Overflow.                   |
| Blackhole  | Blackhole Attack.                      |
| Diversion  | Traffic Diversion Attack.              |


More information about what is the idea behind the dataset, how it is collected, etc. can be found in the [UNR-IDD Intrusion Detection Dataset 2023](https://www.tapadhirdas.com/unr-idd-dataset) website.


## Data Exploration & Experimentation

The whole dataset has 37441 entries. The dataset is cleared and doesn't have any duplicates(it had only one duplicate in the original dataset). The dataset is imbalanced. There are some switches that don't have normal data. We run experiments with both all switches and only with switches that have balanced data with normal network data. 

For centralized learning we are doing Leave-One-Subject-Out Cross Validation, mostly because the dataset is small and to have a compariable results to the federated learning experiments, because they are done on client level(switch). For the centralized learning also deep learning approaches like TabNet were tested, but the results were not as good as the XGBoost model, that's why the experimnets with federated learning are done only with XGBoost model.

For federated learning we are doing evaluation on the client level. Each client(switch) dataset is divided on train/test set(80:20 ratio) by keeping the same distribution of the classes. We tried different combination of rounds and number of clients for the federated learning experiments. The algorithm used for the federated learning is FedXgbBagging.

The main dataset which results are used has these clients(switches):
["of_000000000000000c", "of_000000000000000a", "of_000000000000000b", "of_0000000000000003","of_0000000000000004"]


## Results

The results of the models can be seen by running the mlflow server in the pipeline folder. To run the mlflow server issue the following command in the terminal:

```bash
mlflow server
```

The results of the models can be seen in the mlflow UI. The mlflow UI can be accessed at http://localhost:5000.


## Repository Structure

```
.
├── .vscode                                         # VS Code settings
├── src                                             # Source files
│   ├── dataset                                     # Dataset files
│   │   ├── initial_dataset                         # Initial dataset (raw)
│   │   └── processed_dataset                       # Processed dataset
│   ├── federated_pipeline                          # Federated learning pipeline files
│   │   ├── xgboost_federated_client.py             # XGBoost federated client
│   │   ├── xgboost_federated_server.py             # XGBoost federated server
│   │   └── xgboost_federated.py                    # XGBoost federated learning pipeline
│   ├── pipeline                                    # Centralized pipeline
│   │   ├── centralized_inference_pipeline.py       # Centralized inference pipeline
│   │   └── data_pipeline.py                        # Data preparation pipeline
│   └── utils                                       # Utility files
│       ├── centralized_inferences.py               # Centralized inferences class
│       ├── data_preparation.py                     # Data preparation functions
│       └── validation.py                           # Validation functions
├──.gitignore                                       # Gitignore file
├── README.md                                       # This file
└── requirements.txt                                # Requirements file
```