import ale_py
import gymnasium as gym
import numpy as np
import os
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict

from .model import DecisionTransformer


gym.register_envs(ale_py)

def test(config: dict):
    act_dim = config['act_dim']
    embed_size = config['embed_size']
    game = config['game']
    game_version = config['game_version']
    n_heads = config['n_heads']
    n_layers = config['n_layers']
    max_context_length = config['max_context_length']
    max_steps = config['max_steps']
    model_dir = config['model_dir']
    model_name = config['model_name']
    num_episodes = config['num_episodes']
    state_dim = config['state_dim']
    target_return = config['target_return']

    # Load the environment
    game_name = f"{game}-{game_version}"
    env = gym.make(game_name, obs_type="grayscale")
    
    # Load the model
    model = DecisionTransformer(
        embed_size=embed_size,
        max_context_length=max_context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    
    # Load the saved weights
    state_dict = safe_load(os.path.join(model_dir, model_name))
    load_state_dict(model, state_dict)
    
    total_rewards = []

    Tensor.training = False
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        step = 0
        
        R = [target_return]
        s = [state.flatten()]
        a = []
        t = [1]
        
        while not done and step < max_steps:
            states_tensor = Tensor(np.array(s)).unsqueeze(0)
            actions_tensor = Tensor(np.array(a + [0] * (len(s) - len(a)))).unsqueeze(0).unsqueeze(-1)
            returns_to_go_tensor = Tensor(np.array(R)).unsqueeze(0).unsqueeze(-1)
            timesteps_tensor = Tensor(np.array(t)).unsqueeze(0).unsqueeze(-1)

            if states_tensor.shape[1] < max_context_length:
                pad_length = max_context_length - states_tensor.shape[1]
                states_tensor = states_tensor.pad((None, (0, pad_length), None))
                actions_tensor = actions_tensor.pad((None, (0, pad_length), None))
                returns_to_go_tensor = returns_to_go_tensor.pad((None, (0, pad_length), None))
                timesteps_tensor = timesteps_tensor.pad((None, (0, pad_length), None))
                
            action_preds = model(states_tensor, actions_tensor, returns_to_go_tensor, timesteps_tensor)
            action = action_preds[0, -1].argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            R.append(R[-1] - reward)  # decrement returns-to-go with reward
            s.append(next_state.flatten())
            a.append(action)
            t.append(len(R))
            
            R = R[-max_context_length:]
            s = s[-max_context_length:]
            a = a[-max_context_length:]
            t = t[-max_context_length:]
            
            state = next_state
            step += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")
    
    return avg_reward
