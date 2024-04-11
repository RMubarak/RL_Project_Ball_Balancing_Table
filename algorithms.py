import gym
from collections import defaultdict
import numpy as np
from typing import Callable, Tuple,Optional
from env import  *
from policy import *
import matplotlib.pyplot as plt

# MC
def on_policy_mc_control_epsilon_soft(env: gym.Env, num_episodes: int, gamma: float, epsilon: float):
    final_epoch = []

    name = "MC on policy epsilon soft"

    for j in range(10):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        N = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        episode = []
        state = env.reset()
        epoch_counter = []
        for i in range(num_episodes):
            if episode == []:
                action = env.action_space.sample()
            else:
                action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state,action,reward))
            if done or len(episode)==459:
                G=0
                visited_states =[]
                for t in range(len(episode)-1,-1,-1):
                    state,action,reward = episode[t]
                    G = gamma*G+reward
                    pair=(state,action)
                    if pair  not in visited_states:
                        visited_states.append(pair)
                        N[state][action]+=1
                        Q[state][action] += (G-Q[state][action])/N[state][action]

                episode = []
                state = env.reset()
                epoch_counter.append(epoch_counter[-1]+1)
            else:
                state = next_state
                if epoch_counter == []:
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)
        final_epoch.append(epoch_counter)
    return final_epoch,name

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    final_epoch = []
    name= "SARSA"

    for time_rep in range(10):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        state = env.reset()
        action = policy(state)
        done = False
        epoch_counter = []

        for i in range(num_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            if done:
                epoch_counter.append(epoch_counter[-1]+1)
                state = env.reset()
                action = policy(state)
                done = False
            else:
                if (epoch_counter ==[]):
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)

                state = next_state
                action = next_action

        final_epoch.append(epoch_counter)
    return final_epoch, name

# N Step SARSA
def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    final_epoch = []
    name="n-Step SARSA"

    for k in range(10):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q,epsilon=epsilon)
        step =0
        n=4

        epoch_counter=[]
        state = env.reset()
        action = policy(state)
        T = float('inf')
        t=0

        rewards=[0]
        actions=[action]
        states=[state]
        tau =0
        epoch_counter=[]

        while step<num_steps:
            if t<T:
                next_state, reward, done, _= env.step(action)
                rewards.append(reward)
                states.append(next_state)

                if done:
                    T = t+1
                else:
                    next_action = policy(next_state)
                    actions.append(next_action)

            tau = t-n+1

            if tau>=0:
                G = sum([gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])
                if tau + n < T:
                    G += gamma**n * Q[states[tau + n]][actions[tau + n]]

                Q[states[tau]][actions[tau]] += step_size * (G - Q[states[tau]][actions[tau]])

            if (tau == T-1):
                state = env.reset()
                action = policy(state)
                T = float('inf')
                t= 0
                tau = 0

                rewards=[0]
                actions=[action]
                states=[state]
                step+=1
                epoch_counter.append(epoch_counter[-1]+1)

            else:
                state = next_state
                action = next_action
                t += 1
                step+=1
                if epoch_counter==[]:
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)

        final_epoch.append(epoch_counter)
    return final_epoch, name

# Expected SARSA
def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    final_epoch=[]
    name="expected SARSA"

    for time_rep in range(10):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        state = env.reset()
        action = policy(state)
        done = False
        epoch_counter=[]

        for i in range(num_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)

            expected_q=0
            q_max = np.max(Q[next_state])
            greedy_actions =0

            for i in range(env.action_space.n):
                if Q[next_state][i]==q_max:
                    greedy_actions += 1

            non_greedy_action_probability = epsilon / env.action_space.n
            greedy_action_probability = ((1 - epsilon) / greedy_actions) + non_greedy_action_probability

            for i in range(env.action_space.n):
                if Q[next_state][i] == q_max:
                    expected_q += Q[next_state][i] * greedy_action_probability
                else:
                    expected_q += Q[next_state][i] * non_greedy_action_probability

            Q[state][action] += step_size * (reward + gamma * expected_q - Q[state][action])

            if done:
                epoch_counter.append(epoch_counter[-1]+1)
                state = env.reset()
                action = policy(state)
                done = False
            else:
                if (epoch_counter ==[]):
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)

                state = next_state
                action = next_action

        final_epoch.append(epoch_counter)
    return final_epoch,name

# Q-Learning
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
    final_epoch=[]
    name = "Q-Learning"

    for k in range(10):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = create_epsilon_policy(Q, epsilon)
        epoch_counter=[]
        state = env.reset()
        done = False

        for step in range(num_steps):
            action=policy(state)
            next_state, reward, done, _ = env.step(action)
            next_action = np.max(Q[state])

            max_action = np.argwhere(Q[state]==next_action).flatten()
            next_action = np.random.choice(max_action)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            if done:
                epoch_counter.append(epoch_counter[-1]+1)
                state = env.reset()
                done = False
            else:
                if (epoch_counter ==[]):
                    epoch_counter.append(0)
                else:
                    epoch_counter.append(epoch_counter[-1]+0)
                state = next_state

        final_epoch.append(epoch_counter)
    return final_epoch,name

def get_Q_val(env: gym.Env,num_steps: int,gamma: float,epsilon: float,step_size: float):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    done = False

    for step in range(num_steps):
            action=policy(state)
            next_state, reward, done, _ = env.step(action)

            next_action = np.max(Q[next_state])
            max_action = np.argwhere(Q[next_state]==next_action).flatten()
            next_action = np.random.choice(max_action)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            if done:
                state = env.reset()
                done = False
            else:
                state = next_state
    return Q

def rollout(eps, Q, env):
    episode = []
    for  i in range(eps):
        states,actions,rewards =[],[],[]
        state = env.reset()
        policy =create_epsilon_policy(Q,epsilon=0.1)
        done = False

        while not done:
            action = policy(state)
            states.append(state)
            actions.append(action)

            next_state, reward, done, _ = env.step(action=action)
            rewards.append(reward)
            state=next_state

        episode.append((states,actions,rewards))
    return episode

def td_prediction(env: gym.Env, gamma: float, episodes, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    V = defaultdict(float)
    learning_targets = []

    for episode in episodes:
        states, actions, rewards = episode

        T = len(states)
        for t in range(T):
            G_t = sum(gamma ** (i - t - 1) * rewards[i] for i in range(t, min(t + n,len(states))))

            if t + n < T:
                G_t += gamma ** n * V[states[t + n]]

            V[states[t]] += 0.5*(G_t - V[states[t]])
            if states[t]==(0, 3):
                learning_targets.append(V[states[t]])

    return V, learning_targets

def monte_carlo_prediction(episodes: List[Tuple[List[int], List[int], List[float]]], gamma: float):
    V = defaultdict(float)
    N = defaultdict(int)
    learning_targets = []

    for episode in episodes:
        G = 0
        states, _, rewards = episode
        visited_states = []

        for t in range(len(states) - 1, -1, -1):
            state,reward = states[t],rewards[t]
            G = gamma * G + reward

            if state not in visited_states:
                N[state] = N[state]+1
                V[state] = V[state] + (G-V[state])/N[state]
                visited_states.append(state)

            if state == (0,3):
                learning_targets.append(V[state])

    return V, learning_targets

def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    targets = np.zeros(len(episodes))

    pass
