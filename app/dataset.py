from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import gzip
import numpy as np
from numpy._typing import ArrayLike
import os
import pickle
from typing import Generator, Tuple

from .buffer import ReplayBuffer


class DQNDataset:
    def __init__(
        self,
        state_dim: int,
        max_timesteps: int,
        game: str,
        max_context_length: int,
        data_dir: str = "data",
        size: int = 500_000,
        split: int = 1,
        num_checkpoints: int = 50,
        save_dir: str = "data",
        max_concurrent: int = 4,
    ):
        self.max_context_length = max_context_length
        self.state_dim = state_dim
        self.max_timesteps = max_timesteps
        self.game = game
        self.data_dir = data_dir
        self.size = size
        self.split = str(split)
        self.num_checkpoints = num_checkpoints
        self.save_dir = save_dir
        self.max_concurrent = max_concurrent

        self.filename = f"DQNDataset_{game}_{state_dim}_{max_timesteps}_{size}_{split}_{num_checkpoints}.pkl"
        self.filepath = os.path.join(save_dir, self.filename)

        if os.path.exists(self.filepath):
            print(f"Loading existing dataset from {self.filepath}")
            self.load()
        else:
            print(f"Creating new dataset and saving to {self.filepath}")
            self.create_dataset()
            self.save()

    def create_dataset(self):
        size_per_buffer = self.size // self.num_checkpoints
        checkpoints = 50 - np.random.choice(
            np.arange(self.num_checkpoints), self.num_checkpoints, replace=False
        )

        self.states = np.zeros((self.size, self.state_dim), dtype=np.uint8)
        self.actions = np.zeros((self.size, 1), dtype=np.int8)
        self.returns_to_go = np.zeros((self.size, 1), dtype=np.float32)
        self.timesteps = np.zeros((self.size, 1), dtype=np.int32)
        self.done_idx = []

        total_trajectories = 0
        current_index = 0

        for i in range(0, len(checkpoints), self.max_concurrent):
            chunk = checkpoints[i : i + self.max_concurrent]

            with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                future_to_ckpt = {
                    executor.submit(
                        self.process_checkpoint, ckpt, size_per_buffer
                    ): ckpt
                    for ckpt in chunk
                }
                for future in as_completed(future_to_ckpt):
                    ckpt = future_to_ckpt[future]
                    buffer, num_trajectories = future.result()

                    buffer_size = len(buffer)
                    end_index = current_index + buffer_size

                    self.states[current_index:end_index] = buffer.states
                    self.actions[current_index:end_index] = buffer.actions
                    self.returns_to_go[current_index:end_index] = buffer.returns_to_go
                    self.timesteps[current_index:end_index] = buffer.timesteps

                    # Correct the done_idx values and extend the list
                    self.done_idx.extend(np.array(buffer.done_idx) + current_index)

                    current_index = end_index
                    total_trajectories += num_trajectories
                    print(
                        f"Checkpoint {ckpt} processed. Total trajectories: {total_trajectories}"
                    )

                del buffer
                gc.collect()

        # Trim arrays to actual size used
        self.states = self.states[:current_index]
        self.actions = self.actions[:current_index]
        self.returns_to_go = self.returns_to_go[:current_index]
        self.timesteps = self.timesteps[:current_index]

        # Convert done_idx to numpy array
        self.done_idx = np.array(self.done_idx, dtype=np.int32)

        self.total_size = current_index

        print(f"Dataset created. Total size: {self.total_size}")

    def process_checkpoint(self, ckpt, size_per_buffer):
        buffer = ReplayBuffer(size_per_buffer, self.state_dim)

        obs_file = os.path.join(
            self.data_dir,
            self.game,
            self.split,
            "replay_logs",
            f"$store$_observation_ckpt.{ckpt}.gz",
        )
        action_file = os.path.join(
            self.data_dir,
            self.game,
            self.split,
            "replay_logs",
            f"$store$_action_ckpt.{ckpt}.gz",
        )
        reward_file = os.path.join(
            self.data_dir,
            self.game,
            self.split,
            "replay_logs",
            f"$store$_reward_ckpt.{ckpt}.gz",
        )
        terminal_file = os.path.join(
            self.data_dir,
            self.game,
            self.split,
            "replay_logs",
            f"$store$_terminal_ckpt.{ckpt}.gz",
        )

        with gzip.open(obs_file, "rb") as f:
            observations = np.load(f).reshape(-1, 84 * 84)
        with gzip.open(action_file, "rb") as f:
            actions = np.load(f).reshape(-1, 1)
        with gzip.open(reward_file, "rb") as f:
            rewards = np.load(f).reshape(-1, 1)
        with gzip.open(terminal_file, "rb") as f:
            terminals = np.load(f).reshape(-1, 1)

        episode_starts = np.where(terminals)[0] + 1
        episode_starts = np.concatenate(([0], episode_starts))

        num_trajectories = 0
        for start, end in zip(episode_starts[:-1], episode_starts[1:]):
            end = min(end, start + self.max_timesteps)
            episode_length = end - start

            states = observations[start:end]
            episode_actions = actions[start:end]
            episode_rewards = rewards[start:end]
            returns_to_go = np.cumsum(episode_rewards[::-1])[::-1].reshape(-1, 1)

            buffer.add(
                states=states,
                actions=episode_actions,
                returns_to_go=returns_to_go,
                timesteps=np.arange(start, end).reshape(-1, 1),
                done_idx=episode_length - 1,
            )

            num_trajectories += 1

            if len(buffer) >= size_per_buffer:
                break

        return buffer, num_trajectories

    def __getitem__(
        self, index: int
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        done_idx_idx = np.searchsorted(self.done_idx, index, side="right")

        done_idx = self.done_idx[done_idx_idx]
        done_idx = min(done_idx, index + self.max_context_length)

        index = done_idx - self.max_context_length

        states = self.states[index:done_idx]
        actions = self.actions[index:done_idx]
        returns_to_go = self.returns_to_go[index:done_idx]
        timesteps = self.timesteps[index:done_idx]

        start_idx = 0 if done_idx_idx == 0 else self.done_idx[done_idx_idx - 1]
        start_idx = max(start_idx, index - 4)

        # Reshape states to include 4 frames
        reshaped_states = []
        frames = deque(maxlen=4)
        for i in range(4):
            idx = max(index - 4 + i, start_idx)
            frames.append(self.states[idx])

        for state in states:
            frames.append(state)
            reshaped_states.append(np.stack(frames).flatten())

        reshaped_states = np.array(reshaped_states)

        return reshaped_states, actions, returns_to_go, timesteps

    def __len__(self) -> int:
        return self.total_size

    def batches(
        self, batch_size: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        indices = np.random.permutation(self.total_size)
        for start_idx in range(0, self.total_size, batch_size):
            end_idx = min(start_idx + batch_size, self.total_size)
            batch_indices = indices[start_idx:end_idx]

            batch_states = []
            batch_actions = []
            batch_returns = []
            batch_timesteps = []

            for idx in batch_indices:
                states, actions, returns, timesteps = self[idx]

                batch_states.append(states)
                batch_actions.append(actions)
                batch_returns.append(returns)
                batch_timesteps.append(timesteps)

            yield (
                np.array(batch_states),
                np.array(batch_actions),
                np.array(batch_returns),
                np.array(batch_timesteps),
            )

    def save(self):
        data = {
            "states": self.states,
            "actions": self.actions,
            "returns_to_go": self.returns_to_go,
            "timesteps": self.timesteps,
            "done_idx": self.done_idx,
            "total_size": self.total_size,
            "state_dim": self.state_dim,
            "max_context_length": self.max_context_length,
            "max_timesteps": self.max_timesteps,
            "game": self.game,
            "size": self.size,
            "split": self.split,
            "num_checkpoints": self.num_checkpoints,
        }

        with open(self.filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Dataset saved to {self.filepath}")

    def load(self):
        with open(self.filepath, "rb") as f:
            data = pickle.load(f)

        self.states = data["states"]
        self.actions = data["actions"]
        self.returns_to_go = data["returns_to_go"]
        self.timesteps = data["timesteps"]
        self.done_idx = data["done_idx"]
        self.total_size = data["total_size"]
        self.state_dim = data["state_dim"]
        self.max_context_length = data["max_context_length"]
        self.max_timesteps = data["max_timesteps"]
        self.game = data["game"]
        self.size = data["size"]
        self.split = data["split"]
        self.num_checkpoints = data["num_checkpoints"]

        print(f"Dataset loaded from {self.filepath}")
