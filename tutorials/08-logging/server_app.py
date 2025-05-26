"""myapp: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from myapp.task import Net, get_weights, set_weights, test, get_transforms
from myapp.my_strategy import CustomFedAvg
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader



def get_evaluate_fn(testloader, device):
    
    def evaluate(server_round, parameters_ndarrays, config):
        
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        
        return loss, {"cen_accuracy": accuracy}
    
    return evaluate
        
        
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute accuracies x examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"weighted accuracy": sum(accuracies) / total_examples}

    """
    The weighted_average function calculates
    a weighted average of accuracy metrics from
    all participating clients, where each client's
    contribution is proportional to the size of their dataset.
    """
    
def handle_metrics_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    for num_examples, m in metrics:
        print(f"Client with {num_examples} examples returned metrics: {m}")
    
    # Optionally return something meaningful (or an empty dict)
    return {}

def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learning rate based on current round."""
    lr = 0.01
    # Appply a simple learning rate decay
    if server_round > 2:
        lr = 0.005
    return {"lr": lr}



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # load global test set
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
    
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)
    
    
    
    
    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_metrics_fn,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu")
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
