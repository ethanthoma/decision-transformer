import numpy as np
from numpy._typing import ArrayLike
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim

        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, 1))
        self.returns_to_go = np.zeros((capacity, 1))
        self.timesteps = np.zeros((capacity, 1))
        self.done_idx = []

        self.index = 0
        self.full = False

    def add(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        returns_to_go: ArrayLike,
        timesteps: ArrayLike,
        done_idx: int,
    ) -> None:
        if self.full:
            raise ValueError("Cannot add to buffer, it is full.")

        length = min(states.shape[0], self.capacity - self.index)

        # Pad arrays if they're too short
        def pad_array(arr, target_length):
            if arr.shape[0] < target_length:
                pad_width = ((0, target_length - arr.shape[0]), (0, 0))
                return np.pad(arr, pad_width, mode="constant")
            return arr[:target_length]

        states = pad_array(states, length)
        actions = pad_array(actions, length)
        returns_to_go = pad_array(returns_to_go, length)
        timesteps = pad_array(timesteps, length)

        self.states[self.index : self.index + length] = states
        self.actions[self.index : self.index + length] = actions
        self.returns_to_go[self.index : self.index + length] = returns_to_go
        self.timesteps[self.index : self.index + length] = timesteps
        self.done_idx.append(min(done_idx, length) + self.index)
        self.index = self.index + length

        if self.index == 0:
            self.full = True

    def __len__(self) -> int:
        return self.index
