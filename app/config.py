import argparse

def get_config():
    parser = argparse.ArgumentParser(description="RL training and testing")

    subparsers = parser.add_subparsers(dest='command', help='Subcommands for training and testing')

    # Training parameters
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset_size', type=int, default=1_000, help="Dataset size")
    train_parser.add_argument('--dataset_dir', type=str, default="data", help="DQN Replay dataset directory")
    train_parser.add_argument('--max_timesteps', type=int, default=1_000, help="Maximum timesteps")
    train_parser.add_argument('--embed_size', type=int, default=128, help="Embedding size")
    train_parser.add_argument('--n_heads', type=int, default=8, help="Number of heads in the model")
    train_parser.add_argument('--n_layers', type=int, default=6, help="Number of layers in the model")
    train_parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    train_parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    train_parser.add_argument('--lr', type=float, default=6e-4, help="Learning rate")
    train_parser.add_argument('--num_checkpoints', type=int, default=50, help="Number of checkpoints to load")
    train_parser.add_argument('--split', type=int, default=1, help="Split number to load")
    train_parser.add_argument('--train_set_path', type=str, default="train", help="Path to save the training set")

    # Testing parameters
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--embed_size', type=int, default=128, help="Embedding size")
    test_parser.add_argument('--n_layers', type=int, default=6, help="Number of layers in the model")
    test_parser.add_argument('--n_heads', type=int, default=8, help="Number of heads in the model")
    test_parser.add_argument('--num_episodes', type=int, default=10, help="Number of episodes")
    test_parser.add_argument('--max_steps', type=int, default=1000, help="Maximum steps per episode")
    test_parser.add_argument('--model_name', type=str, default="model-epoch5.safetensors", help="Name of the trained model")
    test_parser.add_argument('--game_version', type=str, default="v4", help="Game version")
    test_parser.add_argument('--target_return', type=int, default=20, help="Target return")

    # Common parameters
    for subparser in [train_parser, test_parser]:
        subparser.add_argument('--game', type=str, default="Pong", help="Game to play")
        subparser.add_argument('--max_context_length', type=int, default=50, help="Maximum context length")
        subparser.add_argument('--state_dim', type=int, default=84*84, help="State dimension")
        subparser.add_argument('--act_dim', type=int, default=6, help="Action dimension")
        subparser.add_argument('--model_dir', type=str, default="models", help="Model directory")

    args = parser.parse_args()
    config = vars(args)

    if not config['command']:
        parser.error("Please specify either 'train' or 'test' command")

    return config

