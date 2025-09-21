import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# class ServerModel(nn.Module):
#     # A simple binary classifier (1 neuron + sigmoid) that takes the concatenated embeddings from all clients and predicts survival.
#     def __init__(self, input_size):
#         super(ServerModel, self).__init__()
#         self.fc = nn.Linear(input_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc(x)
#         return self.sigmoid(x)

class ServerModel(nn.Module):
    def __init__(self, num_clients=3, features_per_client=4):
        super().__init__()
        input_size = num_clients * features_per_client
        self.fc1 = nn.Linear(input_size, 16)  # Hidden layer with 8 neurons
        self.relu = nn.ReLU()                # ReLU activation
        self.fc2 = nn.Linear(16, 1)           # Output layer
        self.sigmoid = nn.Sigmoid()          # Sigmoid for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Strategy(fl.server.strategy.FedAvg):
    
    # Normally FedAvg averages weights.
    # Here, it’s customized to work with embeddings instead of raw model weights.
    
    def __init__(self, labels, num_clients=3, features_per_client=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ServerModel(num_clients, features_per_client)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)
        self.criterion = nn.BCELoss()
        self.label = torch.tensor(labels).float().unsqueeze(1)
    
    # def __init__(self, labels, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
        
    #     self.model = ServerModel(12)
    #     # Builds a ServerModel expecting 12 input features (because each client outputs 4 features → 3×4 = 12).
    #     self.initial_parameters = ndarrays_to_parameters(
    #         [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    #     )
    #     self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
    #     self.criterion = nn.BCELoss()
    #     # Creates optimizer & loss function (BCELoss)
        
    #     self.label = torch.tensor(labels).float().unsqueeze(1)
    #     # Stores labels (Survived) from the Titanic dataset.
    
    def aggregate_fit(self, rnd, results, failures):
        if not self.accept_failures and failures:
            return None, {}

        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]
        embeddings_aggregated = torch.cat(embedding_results, dim=1)
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        output = self.model(embedding_server)
        loss = self.criterion(output, self.label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Split gradients dynamically based on number of clients
        num_clients = len(embedding_results)
        grads = embedding_server.grad.split([4] * num_clients, dim=1)  # 4 features per client
        np_grads = [grad.numpy() for grad in grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        with torch.no_grad():
            correct = 0
            output = self.model(embedding_server)
            predicted = (output > 0.5).float()
            correct += (predicted == self.label).sum().item()
            accuracy = correct / len(self.label) * 100

        metrics_aggregated = {"accuracy": accuracy}
        return parameters_aggregated, metrics_aggregated
    
    
        
    # def aggregate_fit(
    #     # Instead of aggregating weights, it:
        
        
    #     self,
    #     rnd,
    #     results, # 1. Gets the embeddings from all clients (results list).
    #     failures,
    # ):
    

        
        
    #     # Do not aggregate if there are failures and failures are not accepted
    #     if not self.accept_failures and failures:
    #         return None, {}

    #     # Convert results
    #     embedding_results = [
    #         torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
    #         for _, fit_res in results
    #     ]
        
    #     # 2. Concatenates them → creates a full feature vector for each passenger.
    #     embeddings_aggregated = torch.cat(embedding_results, dim=1)
    #     # 3. Runs them through the ServerModel → predicts survival probability.
    #     embedding_server = embeddings_aggregated.detach().requires_grad_()
        
    #     output = self.model(embedding_server) # server forward pass
        
    #     # 4. Calculates loss vs. true labels.
    #     loss = self.criterion(output, self.label)
    #     # 5. Backpropagates to get gradients with respect to each client’s embeddings.
    #     loss.backward()

    #     self.optimizer.step()
        
    #     self.optimizer.zero_grad()

    #     # 6. Splits these gradients into 3 parts ([4, 4, 4]) — one per client.
    #     grads = embedding_server.grad.split([4, 4, 4], dim=1)
    #     np_grads = [grad.numpy() for grad in grads]
    #     parameters_aggregated = ndarrays_to_parameters(np_grads)
    #     # 7. Sends the correct gradient back to each client.
    #     with torch.no_grad():
    #         correct = 0
    #         output = self.model(embedding_server)
    #         predicted = (output > 0.5).float()

    #         correct += (predicted == self.label).sum().item()

    #         accuracy = correct / len(self.label) * 100

    #     metrics_aggregated = {"accuracy": accuracy}

    #     return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        return None, {}