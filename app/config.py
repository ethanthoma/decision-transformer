import argparse


def get_config():
    parser = argparse.ArgumentParser(description="RL training and testing")

    subparsers = parser.add_subparsers(
        dest="command", help="Subcommands for training and testing"
    )

    # Training parameters
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--beta_1", type=float, default=0.9, help="AdamW beta 1")
    train_parser.add_argument("--beta_2", type=float, default=0.95, help="AdamW beta 2")
    train_parser.add_argument(
        "--dataset_dir", type=str, default="data", help="DQN Replay dataset directory"
    )
    train_parser.add_argument(
        "--dataset_size", type=int, default=500_000, help="Dataset size"
    )
    train_parser.add_argument(
        "--final_tokens",
        type=int,
        default=2 * 500_000 * 50,
        help="Number of final tokens",
    )
    train_parser.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="Number of concurrent processes",
    )
    train_parser.add_argument(
        "--max_timesteps", type=int, default=100_000, help="Maximum timesteps"
    )
    train_parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    train_parser.add_argument(
        "--num_checkpoints", type=int, default=50, help="Number of checkpoints to load"
    )
    train_parser.add_argument(
        "--save_dir",
        type=str,
        default="data/train",
        help="Training dataset save directory",
    )
    train_parser.add_argument(
        "--split", type=int, default=5, help="Split number to load"
    )
    train_parser.add_argument(
        "--warmup_tokens", type=int, default=20 * 512, help="Number of warmup tokens"
    )
    train_parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay"
    )

    # Testing parameters
    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument(
        "--game_version", type=str, default="v4", help="Game version"
    )
    test_parser.add_argument(
        "--max_steps", type=int, default=1_000, help="Maximum steps per episode"
    )
    test_parser.add_argument(
        "--model_name",
        type=str,
        default="model-epoch5.safetensors",
        help="Name of the trained model",
    )
    test_parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes"
    )
    test_parser.add_argument(
        "--render_mode",
        choices=[None, "human"],
        default=None,
        help="Rendering mode for gymnasium",
    )
    test_parser.add_argument(
        "--target_return", type=int, default=20, help="Target return"
    )

    # Common parameters
    for subparser in [train_parser, test_parser]:
        subparser.add_argument(
            "--act_dim", type=int, default=6, help="Action dimension"
        )
        subparser.add_argument(
            "--embed_size", type=int, default=128, help="Embedding size"
        )
        subparser.add_argument("--game", type=str, default="Pong", help="Game to play")
        subparser.add_argument(
            "--loop", type=bool, default=False, help="Use loop transformer"
        )
        subparser.add_argument(
            "--max_context_length", type=int, default=50, help="Maximum context length"
        )
        subparser.add_argument(
            "--model_dir", type=str, default="models", help="Model directory"
        )
        subparser.add_argument(
            "--n_heads", type=int, default=8, help="Number of heads in the model"
        )
        subparser.add_argument(
            "--n_layers", type=int, default=6, help="Number of layers in the model"
        )
        subparser.add_argument(
            "--state_dim", type=int, default=84 * 84, help="State dimension"
        )

    args = parser.parse_args()
    config = vars(args)

    if not config["command"]:
        parser.error("Please specify either 'train' or 'test' command")

    return config
