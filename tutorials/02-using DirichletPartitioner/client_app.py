"""myapp: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from myapp.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Extract config from Flower
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    # Log or print the values
    print(f"[Client] partition_id: {partition_id}, num_partitions: {num_partitions}, local_epochs: {local_epochs}")

    # Load model and data
    net = Net()
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Return the client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()



# Flower ClientApp
app = ClientApp(
    client_fn,
)
