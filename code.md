Main Function BreakDown:

Parsering:

    - To Add Description -> parser = argparse.ArgumentParser(description="")
    - To Add Argument -> parser.add_argument("--argument", action="", type=value, default=value, help="Description")
    - To read arguments and store in object format -> args = parser.parse_args()

Logging:

    - To create a log file -> setup_logging(log_directory)
    - Format -> "timestamp - logger_name - level - message"
    - To assign logger a name -> logger = logging.getLogger(__name__)

Simulation:

    - Checks for the clients, rounds and the config value.
    - If config file is not provided, opens the config at the default path.
    - Checks the dataset path and the client partitions
    - Initializes Client and Server Functions
    - Run simulation Function with backend config.
    - Stores history in a specified json file with significant arguments.

Functions :

1) validate_s3prl_compatibility:
    - Checks for the s3prl imports
    - Checks for hubert model config (different layer and learning parameters)
    - Checks for the pretraining config (different layer and learning parameters)
    - Checks model creation 

2) server_fn:
    - Load the simulation and pretraining configuration.
    - Load the pseudo labels from the kmeans_target file.
    - Initialize model and task parameters
    - Setup the aggregation strategy
    - 

3) save_checkpoint:
    - Save the model according to the s3prl compatible format

4) load_checkpoint: 
    - Load the model in a s3prl compatible format

Classes:

1) FedAdam:
    - Extends the strategy class provided by flwr framework
    - Validates the arguments provided
    - Ensure all parameters are of type float32 numpy arrays.
    - Check clients parameters for consistency
    - Initialize parameters if not found.
    - configure_fit() -> For next round of training provide proper FitIns Object
    - configure_evaluate() -> For next round of evaluation provide proper EvaluteIns Object
    - aggregate_fit() -> Aggregate the results provided by the clients.
    - aggregate_evaluate() -> Evaluate the results provided by the clients.
    - evaluate() -> Server side evaluation.


