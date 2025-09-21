# This is what Flower runs for each client.

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch

from vfl.task import ClientModel, load_data


class FlowerClient(NumPyClient):
    
    
    def __init__(self, v_split_id, data, lr):
        # v_split_id tells it which vertical slice it owns.
        self.v_split_id = v_split_id
        self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
        # Data is standardized (StandardScaler).
        self.model = ClientModel(input_size=self.data.shape[1])
        # Model = ClientModel (from task.py) that turns its features into a 4D embedding.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        embedding = self.model(self.data)
        return [embedding.detach().numpy()], 1, {}
    
    # Does forward pass on all local data → produces embeddings.
    # Sends embeddings to server

    def evaluate(self, parameters, config):
        self.model.zero_grad()
        embedding = self.model(self.data)
        embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))
        self.optimizer.step()
        return 0.0, 1, {}
    
    # Runs forward pass again.
    # Backpropagates using gradient sent from server (parameters list).
    # Updates its own local model weights.
    # Returns dummy loss/metrics (not used here).


def client_fn(context: Context):
    
    # Flower calls this to make a client instance.
    # Loads the correct vertical & horizontal data partition via load_data()
    # Passes the data + split ID + LR to FlowerClient
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
# Registers this client logic with Flower.