import numpy as np
from numpy._typing import ArrayLike
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, max_context_length: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.max_context_length = max_context_length

        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, 1))
        self.returns_to_go = np.zeros((capacity, 1))
        self.timesteps = np.zeros((capacity, 1))
        self.done_idx = []

        self.finished_adding = False

        self.index = 0

    def add(self, states: ArrayLike, actions: ArrayLike, returns_to_go: ArrayLike, timesteps: ArrayLike, done_idx: int) -> None:
        if self.finished_adding:
            raise ValueError("Cannot add to buffer after sampling")

        length = states.shape[0]

        self.states[self.index:self.index+length] = states
        self.actions[self.index:self.index+length] = actions
        self.returns_to_go[self.index:self.index+length] = returns_to_go
        self.timesteps[self.index:self.index+length] = timesteps
        self.done_idx.append(done_idx)
        self.index = (self.index + length) % self.capacity

    def __getitem__(self, index: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        if not self.finished_adding:
            self.finished_adding = True
            self.done_idx = np.array(self.done_idx)

        closest_idx = self.done_idx[np.searchsorted(self.done_idx, index, side='right') - 1]
        closest_idx = closest_idx if closest_idx - index <= self.max_context_length else self.max_context_length + index

        return (
            self.states[index:closest_idx],
            self.actions[index:closest_idx],
            self.returns_to_go[index:closest_idx],
            self.timesteps[index:closest_idx],
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.done_idx[-1], size=batch_size)
        batches = [self[i] for i in indices]

        max_len = self.max_context_length
        states = np.zeros((batch_size, max_len, self.state_dim))
        actions = np.zeros((batch_size, max_len, 1))
        returns_to_go = np.zeros((batch_size, max_len, 1))
        timesteps = np.zeros((batch_size, max_len, 1))

        for i, batch in enumerate(batches):
            states[i, :len(batch[0])] = batch[0]
            actions[i, :len(batch[1])] = batch[1]
            returns_to_go[i, :len(batch[2])] = batch[2]
            timesteps[i, :len(batch[3])] = batch[3]

        return states, actions, returns_to_go, timesteps
