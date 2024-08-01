import ale_py
import gymnasium as gym
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict

from .model import DecisionTransformer

gym.register_envs(ale_py)

# model parameters
embed_size = 128
n_layers = 6
n_heads = 8

# testing parameters
num_episodes = 10
max_steps = 1000
model_path = "models/model-epoch5.safetensors"

# game parameters
game = "Pong"
game_version = "v4"
max_context_length = 50
state_dim = 210 * 160
act_dim = 6
target_return = 20

def evaluate_model(model_path, num_episodes=10, max_steps=1000):
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
    state_dict = safe_load(model_path)
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

def test():
    evaluate_model(model_path, num_episodes, max_steps)
