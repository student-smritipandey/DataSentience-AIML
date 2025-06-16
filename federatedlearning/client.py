import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from utils import get_dataloaders
import sys
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1

def main():
    if len(sys.argv) < 2:
        print("❌ Please provide client ID: `python3 client.py 0`")
        sys.exit(1)

    client_id = sys.argv[1]
    print(f"✅ Starting client {client_id} on {DEVICE}")

    # Adjust this path if needed
    base_dir = os.path.join(os.path.dirname(__file__), "federatedlearning", "chest_xray")
    train_path = os.path.join(base_dir, f"client_{client_id}", "train")
    test_path = os.path.join(base_dir, f"client_{client_id}", "test")

    trainloader = get_dataloaders(train_path, batch_size=32)
    testloader = get_dataloaders(test_path, batch_size=32)

    model = get_model().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    class FLClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()
            for _ in range(EPOCHS):
                for x, y in trainloader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    optimizer.step()
            return self.get_parameters(config), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            model.eval()
            correct, total, loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss_sum += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            accuracy = correct / total if total > 0 else 0
            avg_loss = loss_sum / len(testloader)
            print(f"📊 Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            return avg_loss, total, {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient())

if __name__ == "__main__":
    main()
