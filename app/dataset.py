import gzip
import numpy as np
from numpy._typing import ArrayLike
import os
from typing import Tuple

from .buffer import ReplayBuffer

class DQNDataset:
    def __init__(self, state_dim: int, max_context_length: int, max_timesteps: int, game: str, data_dir: str = "data", size: int = 500_000, split: int = 1, num_checkpoints: int = 50):
        self.size = size
        self.num_checkpoints = num_checkpoints
        split: str = str(split)

        capacity = size // num_checkpoints
        self.buffers = [ReplayBuffer(capacity, state_dim, max_context_length) for _ in range(num_checkpoints)]

        total_transitions = 0
        num_trajectories = 0
        while total_transitions < size:
            ckpt = np.random.choice(np.arange(num_checkpoints), 1)[0]
            buffer = self.buffers[ckpt]

            if len(buffer) >= capacity:
                continue

            obs_file = os.path.join(data_dir, game, split, "replay_logs", f"$store$_observation_ckpt.{ckpt}.gz")
            action_file = os.path.join(data_dir, game, split, "replay_logs", f"$store$_action_ckpt.{ckpt}.gz")
            reward_file = os.path.join(data_dir, game, split, "replay_logs", f"$store$_reward_ckpt.{ckpt}.gz")
            terminal_file = os.path.join(data_dir, game, split, "replay_logs", f"$store$_terminal_ckpt.{ckpt}.gz")
            
            with gzip.open(obs_file, 'rb') as f:
                obs_data = f.read()

            num_frames = len(obs_data) // state_dim
            observations = np.frombuffer(obs_data[:num_frames * state_dim], dtype=np.uint8).reshape(-1, state_dim)
            
            with gzip.open(action_file, 'rb') as f:
                actions = np.frombuffer(f.read(), dtype=np.int32).reshape(-1, 1)
            
            with gzip.open(reward_file, 'rb') as f:
                rewards = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 1)
            
            with gzip.open(terminal_file, 'rb') as f:
                terminals = np.frombuffer(f.read(), dtype=np.bool_).reshape(-1, 1)

            episode_starts = np.where(terminals)[0] + 1
            episode_starts = np.concatenate(([0], episode_starts))
            
            for start, end in zip(episode_starts[:-1], episode_starts[1:]):
                episode_length = end - start
                if episode_length > max_timesteps:
                    continue
                
                states = observations[start:end]
                episode_actions = actions[start:end]
                episode_rewards = rewards[start:end]
                
                returns_to_go = np.cumsum(episode_rewards[::-1])[::-1].reshape(-1, 1)
                
                for t in range(episode_length):
                    if t + max_context_length > episode_length:
                        break
                    
                    if len(buffer) >= capacity:
                        break

                    buffer.add(
                        states=states[t:t+max_context_length],
                        actions=episode_actions[t:t+max_context_length],
                        returns_to_go=returns_to_go[t:t+max_context_length],
                        timesteps=np.arange(t, t+max_context_length).reshape(-1, 1),
                        done_idx=episode_length,
                    )

                if len(buffer) >= capacity:
                    break

                num_trajectories += 1

            total_transitions = sum(len(buffer) for buffer in self.buffers)
            print(f"Total transitions: {total_transitions}, Trajectories: {num_trajectories}")

        self.total_size = total_transitions

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        buffer_size = self.size // self.num_checkpoints
        buffer_index = index // buffer_size
        local_index = index % buffer_size
        return self.buffers[buffer_index][local_index]

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        buffer_indices = np.random.randint(0, self.num_checkpoints, size=batch_size)
        local_indices = [np.random.randint(0, len(buffer)) for buffer in self.buffers]
        
        states, actions, returns_to_go, timesteps = [], [], [], []
        
        for buffer_idx, local_idx in zip(buffer_indices, local_indices):
            s, a, r, t = self.buffers[buffer_idx][local_idx]
            states.append(s)
            actions.append(a)
            returns_to_go.append(r)
            timesteps.append(t)
        
        return (np.array(states), np.array(actions), 
                np.array(returns_to_go), np.array(timesteps))
