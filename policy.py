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
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state) -> int:
        if np.random.random() < epsilon:
            action = np.random.choice(np.arange(num_actions))
        else:
            try:
                action = argmax(Q[state]) # Only the best actions can  be chosen
            except:
                action = np.random.choice(np.arange(num_actions)) # In case that state was not seen in training play randomly
        return action

    return get_action

def argmax(arr: np.array) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    arr = np.array(arr)
    
    # Get the first index of the max reward
    ind1 = np.argmax(arr)
    
    # The max number in the list
    indices = np.where(arr == arr[ind1])[0]

    
    return np.random.choice(indices)


# Gets the probability the given action is taken in an epsilon greedy policy with given Q values
def action_prob(Q: defaultdict, state: Tuple[int,int], action: int, epsilon: float):
    arr = np.array(Q[state])
    # Get the first index of the max reward
    ind1 = np.argmax(arr)
    # Gets a list of all greedy actions
    greedy_actions = np.where(arr == arr[ind1])[0]
    
    # Always adds the probability of random exploration
    prob = epsilon*(1/len(arr)) 
    
    # Adds the greedy action selection probability
    if action in greedy_actions:
        prob += (1-epsilon)*1/len(greedy_actions)
    
    return prob

# def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
#     """Creates an epsilon soft policy from Q values.

#     A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

#     Args:
#         Q (defaultdict): current Q-values
#         epsilon (float): softness parameter
#     Returns:
#         get_action (Callable): Takes a state as input and outputs an action
#     """

#     num_actions = len(Q[0])
#     def get_action(state: Tuple) -> int:
#         if np.random.random() < epsilon:
#             action = np.random.choice(num_actions)
#         else:
#             try: # In case the state has not been seen before
#                 action = np.max(Q[state])
#                 max_action = np.argwhere(Q[state]==action).flatten()
#                 action = np.random.choice(max_action)
#             except:
#                 action = np.random.choice(num_actions)
#         return action

#     return get_action
