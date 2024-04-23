import pickle
import gym
from policy import create_epsilon_policy
from tqdm import tqdm
from env import BallBalancingTable
import matplotlib.pyplot as plt 

'''
This will play the game using a loaded Q dictionary for num_episodes and and output the win percentage and total rewards of the policy.
You can choose how many episodes the agent will play and what the time limit is to define success.
These policies were trained for 1,000,000 episodes with a time limit of 5 seconds on the table to define success.

Please uncomment one of the lines below num_episode to choose which policy to use
'''

num_episodes = 1000
time_lim = 5
filepath = None

# Q-Learning
filepath = 'Policies/q_learning_5s_L30.pickle'

# # On-Policy Monte Carlo Control
# filepath = 'Policies/on_policy_mc_control.pickle'

# # Expected SARSA
# filepath = 'Policies/exp_sarsa.pickle'



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



with open(filepath, 'rb') as f:
    Q = pickle.load(f)
    
b = BallBalancingTable(time_limit=time_lim)
steps, rewards, win_perc = play_game(b, num_episodes, Q)

print("Win Percentage Over " + str(num_episodes) + " Episodes is: " + str(win_perc*100) + "%")
print("Total Rewards Over " + str(num_episodes) + " Episodes is: " + str(rewards))