import os
import numpy as np

from .fixed_replay_buffer import FixedReplayBuffer


def create_dataset(game, data_dir_prefix, num_buffers=50):
    data_dir = os.path.join(data_dir_prefix, game, '1', 'replay_logs')
    replay_buffer = FixedReplayBuffer(data_dir, replay_suffix=None, observation_shape=(
        84, 84), stack_size=4, update_horizon=1, gamma=0.99, observation_dtype=np.uint8, batch_size=32, replay_capacity=100000)

    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []

    for _ in range(num_buffers):
        transitions = replay_buffer.sample_transition_batch(
            batch_size=replay_buffer.add_count)
        observations.append(transitions[0])
        actions.append(transitions[1])
        rewards.append(transitions[2])
        terminals.append(transitions[3])
        next_observations.append(transitions[4])

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    terminals = np.concatenate(terminals, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)

    return observations, actions, rewards, terminals, next_observations
