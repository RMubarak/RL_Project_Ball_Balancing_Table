import gym
from collections import defaultdict
import numpy as np
from typing import Callable, Tuple,Optional
from env import  *
from policy import *
import matplotlib.pyplot as plt

def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    t = 0
    num_episodes = 0
    steps = []
    rewards_per_episode = []
    episode_reward = 0
    done = False
    state = env.reset()
    
    
    while t < num_steps:    
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        # To get Q(S', Amax)
        greedy_policy = create_epsilon_policy(Q, 0)
        next_action = greedy_policy(next_state)

        new_Q = Q[state][action] + step_size*(reward + gamma*Q[next_state][next_action] - Q[state][action])
        Q[state][action] = new_Q
        policy = create_epsilon_policy(Q, epsilon)
        
        if done:
            rewards_per_episode.append(episode_reward)
            episode_reward = 0
            num_episodes += 1
            state = env.reset()
        else:
            state = next_state
        
        t += 1
        steps.append(num_episodes)

    return steps, rewards_per_episode, Q