import pickle
import gym
from policy import create_epsilon_policy
from tqdm import tqdm
from env import BallBalancingTable
import matplotlib.pyplot as plt 

'''
This will play the game using loaded Q dictionaries for num_episodes and and output the win percentage and total rewards of the policy. It will also plot the play performance over several win conditions. 
You can choose how many episodes the agent will play and what the time limit is to define success.
These policies were trained for 1,000,000 episodes.
To run several policies, please add each filepath to the list of filepaths below.

Please uncomment one of the lines below num_episode to choose which policy to use
'''

num_episodes = 1000
time_lims = [1,2,3,4,5,6,8,10,15,30]
filepaths = {}
title = None
noise = False # Whether you want to play the game with sensor noise or not

# NOTE: Make the dictionary keys equal to the legend labels

# # Q-Learning: 
# filepaths["5s Training"]= 'Policies/q_learning_5s.pickle'
# filepaths["15s Training"]= 'Policies/q_learning_15s.pickle'
# filepaths["60s Training"]= 'Policies/q_learning_60s.pickle'
# title = "Q-Learning Performance Vs Different Time Limits"

# # Expected SARSA
# filepaths["5s Training"]= 'Policies/exp_sarsa_5s.pickle'
# filepaths["15s Training"]= 'Policies/exp_sarsa_15s.pickle'
# filepaths["60s Training"]= 'Policies/exp_sarsa_60s.pickle'
# title = "Exp SARSA Performance Vs Different Time Limits"

# # On-Policy Monte Carlo Control
# filepaths["5s Training"]= 'Policies/mc_control_5s.pickle'
# filepaths["15s Training"]= 'Policies/mc_control_15s.pickle'
# filepaths["60s Training"]= 'Policies/mc_control_60s.pickle'
# title = "MC Control Performance Vs Different Time Limits"

# # Policy Comparison
# filepaths["Q-Learning"]= 'Policies/q_learning_15s.pickle'
# filepaths["MC Control"]= 'Policies/mc_control_60s.pickle'
# filepaths["Exp SARSA"]= 'Policies/exp_sarsa_60s.pickle'
# title = "Algorithm Performance Comparison"

def play_game(
    env: gym.Env,
    tot_episodes: int,
    Q
):
    """Plays the game according to a greedy policy based on a state-action value dictionary, Q

    Args:
        env (gym.Env): a Gym API compatible environment
        tot_episodes (int): Number of episodes to play
        Q: State-Action Value dictionary
    """
    # Creates a greedy policy
    policy = create_epsilon_policy(Q, 0)
    
    # Initializes values to keep track of
    num_episodes = 0
    steps = []
    total_rewards = 0
    done = False
    state = env.reset()
    won = 0.0
    lost = 0.0
    progress_bar = tqdm(total=tot_episodes)
    
    while num_episodes < tot_episodes:    
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        
        # If episode is done check win/loss conditions and reset 
        if done:
            if reward < 0:
                lost += 1
            else:
                won += 1
        
            num_episodes += 1
            progress_bar.update(1)
            state = env.reset()
        else:
            state = next_state
        
       
        steps.append(num_episodes)

    progress_bar.close()
    # Win Percentage
    win_perc = won/(won+lost)
    return steps, total_rewards,win_perc



for key in filepaths.keys():
    with open(filepaths[key], 'rb') as f:
        Q = pickle.load(f)
    
    tot_rew = []
    tot_win_p = []
    for t in time_lims:
        b = BallBalancingTable(time_limit=t, sensor_noise=noise)
        steps, rewards, win_perc = play_game(b, num_episodes, Q)
        tot_rew.append(rewards)
        tot_win_p.append(win_perc*100)
    plt.figure(1)
    plt.plot(time_lims, tot_win_p, label=key)
    
    plt.figure(2)
    plt.plot(time_lims, tot_rew, label=key)

plt.figure(1)
plt.title(title)
plt.xlabel("Time Limit (s)")
plt.ylabel("Win Percentage Over 1000 Games (%)")    
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.show()

plt.figure(2)
plt.title(title)
plt.xlabel("Time Limit (s)")
plt.ylabel("Total Rewards Over 1000 Games (%)")    
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()