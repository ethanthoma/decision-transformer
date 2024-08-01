import argparse

def get_config():
    parser = argparse.ArgumentParser(description="RL training and testing")

    # Common parameters
    parser.add_argument('--game', type=str, default="Pong", help="Game to play")
    parser.add_argument('--max_context_length', type=int, default=50, help="Maximum context length")
    parser.add_argument('--state_dim', type=int, default=84*84, help="State dimension")
    parser.add_argument('--act_dim', type=int, default=6, help="Action dimension")

    # Training parameters
    parser.add_argument('--train', action='store_true', help="Set this flag for training")
    parser.add_argument('--dataset_size', type=int, default=10_000, help="Dataset size")
    parser.add_argument('--max_timesteps', type=int, default=100_000, help="Maximum timesteps")
    parser.add_argument('--embed_size', type=int, default=128, help="Embedding size")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of heads in the model")
    parser.add_argument('--n_layers', type=int, default=6, help="Number of layers in the model")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=6e-4, help="Learning rate")

    # Testing parameters
    parser.add_argument('--test', action='store_true', help="Set this flag for testing")
    parser.add_argument('--num_episodes', type=int, default=10, help="Number of episodes")
    parser.add_argument('--max_steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--model_path', type=str, default="models/model-epoch5.safetensors", help="Path to the trained model")
    parser.add_argument('--game_version', type=str, default="v4", help="Game version")
    parser.add_argument('--target_return', type=int, default=20, help="Target return")

    args = parser.parse_args()
    config = vars(args)

    if not (config['train'] or config['test']):
        parser.error("Please specify --train or --test")

    return config
