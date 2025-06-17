import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import start_server, ServerConfig

# Define the Federated Averaging strategy
strategy = FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
)

# Start the Flower server
if __name__ == "__main__":
    start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=3),
        strategy=strategy
    )
