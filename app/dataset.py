from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import gzip
import numpy as np
from numpy._typing import ArrayLike
import os
import pickle

from itertools import islice
from typing import Tuple

from .buffer import ReplayBuffer


class DQNDataset:
    def __init__(
        self,
        state_dim: int,
        max_context_length: int,
        max_timesteps: int,
        game: str,
        data_dir: str = "data",
        size: int = 500_000,
        split: int = 1,
        num_checkpoints: int = 50,
        save_dir: str = "data",
        max_concurrent: int = 4,
    ):
        self.state_dim = state_dim
        self.max_context_length = max_context_length
        self.max_timesteps = max_timesteps
        self.game = game
        self.data_dir = data_dir
        self.size = size
        self.split = str(split)
        self.num_checkpoints = num_checkpoints
        self.save_dir = save_dir
        self.max_concurrent = max_concurrent

        self.filename = f"DQNDataset_{game}_{state_dim}_{max_context_length}_{max_timesteps}_{size}_{split}_{num_checkpoints}.pkl"
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
        max_concurrent = 2  # Adjust based on your system

        for i in range(0, len(checkpoints), max_concurrent):
            chunk = checkpoints[i : i + max_concurrent]

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
        closest_idx = self.done_idx[np.searchsorted(self.done_idx, index, side="right")]

        closest_idx = (
            closest_idx
            if closest_idx - index <= self.max_context_length
            else self.max_context_length + index
        )

        states = self.states[index:closest_idx]
        actions = self.actions[index:closest_idx]
        returns_to_go = self.returns_to_go[index:closest_idx]
        timesteps = self.timesteps[index:closest_idx]

        # Reshape states to include 4 frames
        reshaped_states = []
        for i in range(len(states)):
            if i < 3:
                frames = [states[0]] * (3 - i) + list(states[: i + 1])
            else:
                frames = states[i - 3 : i + 1]

            reshaped_frame = np.concatenate(frames, axis=0)
            reshaped_states.append(reshaped_frame)

        reshaped_states = np.array(reshaped_states)

        return reshaped_states, actions, returns_to_go, timesteps

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.done_idx[-1], size=batch_size)
        batches = [self[i] for i in indices]

        max_len = self.max_context_length
        states = np.zeros((batch_size, max_len, 4 * self.state_dim), dtype=np.uint8)
        actions = np.zeros((batch_size, max_len, 1), dtype=np.int8)
        returns_to_go = np.zeros((batch_size, max_len, 1), dtype=np.float32)
        timesteps = np.zeros((batch_size, max_len, 1), dtype=np.int32)

        for i, batch in enumerate(batches):
            states[i, : len(batch[0])] = batch[0]
            actions[i, : len(batch[1])] = batch[1]
            returns_to_go[i, : len(batch[2])] = batch[2]
            timesteps[i, : len(batch[3])] = batch[3]

        return states, actions, returns_to_go, timesteps

    def __len__(self) -> int:
        return self.total_size

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
