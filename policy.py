import numpy as np
from collections import defaultdict
from typing import Callable, Tuple
import gym
from enum import IntEnum
from env import *

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """

    num_actions = len(Q[0])
    def get_action(state: Tuple) -> int:
        if np.random.random() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.max(Q[state])
            max_action = np.argwhere(Q[state]==action).flatten()
            action = np.random.choice(max_action)
        return action

    return get_action
