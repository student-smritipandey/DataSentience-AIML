# Federated Learning (FL) for Chest X-Ray CNN Classification

This module implements a **federated learning pipeline** using the Flower framework to train a convolutional neural network (CNN) on the Kaggle Chest X-ray dataset in a privacy-preserving, multi-client setup.

---

## 🔍 Overview

This is **one component** of the broader **DataSentience-AIML** project submitted for SSoC 2025. The federated learning module showcases how healthcare imaging data (such as chest X-rays) can be used to train AI models without aggregating raw data into a central server. Instead, training occurs locally on multiple simulated clients, and only the model weights are shared and aggregated.

---

## ✅ Key Features

* **Federated Averaging (FedAvg)** strategy using Flower
* Simulated multiple client environment using chest X-ray data
* Training of a CNN without centralizing data
* PyTorch used for model building and training
* Configurable number of FL rounds and clients

---

## 📁 Project Components

* `client.py`: Local client code, handles training on client-specific data
* `server.py`: Server logic that coordinates FL rounds and aggregates weights
* `model.py`: CNN architecture used for pneumonia detection
* `utils.py`: Functions for data preprocessing, loading, and splitting
* `chest_xray/`: Folder containing distributed dataset for clients (from Kaggle dataset)

---

## 🏃‍♀️ How to Run This Module

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start Federated Server

```bash
python server.py
```

### Start Clients in Separate Terminals

```bash
python client.py 0
python client.py 1
```

Each client will train on its own portion of the data.

---

## 🔬 Dataset Used

* **Name**: Chest X-Ray Pneumonia Dataset
* **Source**: Kaggle
* **Classes**: `NORMAL`, `PNEUMONIA`
* Each client gets a separate train/test split of the data.

---

## 🔒 Why Federated Learning?

* Enables model training **without moving sensitive medical data**
* Prevents data leakage and privacy concerns
* Mirrors real-world scenarios (e.g., hospitals collaborating)

---

## 📌 Notes

* This module is self-contained and can be tested independently
* TensorBoard logging is disabled for now to keep the setup simple
* Flower deprecated `start_server()` and `start_client()` — future versions should migrate to the `flower-superlink` and `flower-supernode` CLIs

---

## 📎 License

This FL module is part of the open-source **DataSentience-AIML** project under MIT License
