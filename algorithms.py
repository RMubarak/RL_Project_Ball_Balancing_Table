import gym
from collections import defaultdict
import numpy as np
from typing import Callable, Tuple,Optional
from env import  *
from policy import *
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def q_learning(
    env: gym.Env,
    tot_episodes: int,
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
    won = 0.0
    lost = 0.0
    progress_bar = tqdm(total=tot_episodes)
    while num_episodes < tot_episodes:    
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
            if reward < 0:
                lost += 1
            else:
                won += 1
            
            rewards_per_episode.append(episode_reward)
            episode_reward = 0
            num_episodes += 1
            progress_bar.update(1)
            state = env.reset()
        else:
            state = next_state
        
        t += 1
        steps.append(num_episodes)

    progress_bar.close()
    # Win Percentage
    win_perc = won/(won+lost)
    return steps, rewards_per_episode, Q, win_perc

def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)

    returns = []
    steps = []
    episodes = []
    
    for _ in trange(num_episodes, desc="Episode", leave=False):
        episode = generate_episode(env, policy) # Ends each episode after 459 steps
        episodes.append(episode)
        steps.append(len(episode))
        G = 0
        episode_sa_pairs = [(sar[0],sar[1]) for sar in episode] # Gets all the state-action pairs in the episode
        
        for t in range(len(episode) - 1, -1, -1):
            sar = episode[t]
            state = sar[0]
            action = sar[1]
            reward = sar[2]
            G = gamma*G + reward
            
            # Update V and N here according to first visit MC
            if (state,action) not in episode_sa_pairs[0:t]:    
                N[state][action] += 1
                
                new_Q = Q[state][action] + (1/N[state][action])*(G-Q[state][action])
                Q[state][action] = new_Q
                policy = create_epsilon_policy(Q, epsilon)
                
        returns.append(G)

    return Q

def generate_episode(env: gym.Env, policy: Callable, es: bool = False, limited: bool=False, max_steps: int=0):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    steps = 0
    while True:
        if limited and steps >= max_steps: # If we want to terminate an episode after a certain # of steps
            break
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        steps += 1
        if done:
            break
        state = next_state

    return episode


def exp_sarsa(
    env: gym.Env,
    tot_episodes: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        tot_episodes (int): Number of episodes to play
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # Initializations
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    t = 0
    num_episodes = 0
    steps = []
    done = False
    state = env.reset()
    policy = create_epsilon_policy(Q, epsilon)
    actions = np.arange(env.action_space.n)
    progress_bar = tqdm(total=tot_episodes)
    
    while num_episodes < tot_episodes:    
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        
        # To get expectation of policy in the next state
        exp_next_state = 0
        for act in actions:
            prob = action_prob(Q, next_state, act, epsilon)
            exp_next_state += prob*Q[next_state][act]

        new_Q = Q[state][action] + step_size*(reward + gamma*exp_next_state - Q[state][action])
        Q[state][action] = new_Q
        policy = create_epsilon_policy(Q, epsilon)
        
        if done:
            num_episodes += 1
            progress_bar.update(1)
            state = env.reset()
        else:
            state = next_state
        
        t += 1
        steps.append(num_episodes)
    progress_bar.close()
    return Q