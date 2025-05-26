from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from myapp.task import Net, set_weights
import torch


class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes | BaseException]],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        # Instantiate model
        model = Net()
        set_weights(model, ndarrays)

        # Save global model in the standard PyTorch way
        torch.save(model.state_dict(), f"global_model_round_{server_round}")

        return parameters_aggregated, metrics_aggregated
