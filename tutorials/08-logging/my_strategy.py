from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import torch
import json

from .task import Net, set_weights


class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    saving global checkpoints and saving metrics to the local file system as a JSON.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate received model updates and metrics, save global model checkpoint."""

        # Call the default aggregate_fit method from FedAvg
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        # Instantiate model
        model = Net()

        # Apply parameters to model
        set_weights(model, ndarrays)

        # Save global model in the standard PyTorch way
        torch.save(model.state_dict(), f"global_model_round_{server_round}.pt")

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model and save metrics to local JSON."""
        # Call the default behaviour from FedAvg
        loss, metrics = super().evaluate(server_round, parameters)

        # Store metrics as dictionary
        my_results = {"loss": loss, **metrics}

        # Insert into local dictionary
        self.results_to_save[server_round] = my_results

        # Save metrics as JSON
        with open("results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        return loss, metrics
