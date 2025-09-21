from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vfl.strategy import Strategy
from vfl.task import process_dataset


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Get dataset 
    # Loads processed Titanic data via process_dataset() — so it has the labels.
    processed_df, _ = process_dataset()

    # Define the strategy
    # Creates your custom Strategy with the labels.
    strategy = Strategy(processed_df["Survived"].values)

    # Construct ServerConfig
    # Configures Flower’s server for the number of rounds specified in pyproject.toml.
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    # Returns these as ServerAppComponents to Flower.
    return ServerAppComponents(strategy=strategy, config=config)


# Start Flower server
# makes this runnable as the server.
app = ServerApp(server_fn=server_fn)