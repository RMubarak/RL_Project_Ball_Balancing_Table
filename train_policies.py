from env import BallBalancingTable
import matplotlib.pyplot as plt
from env import  *
from policy import *
import matplotlib.pyplot as plt
import pickle
from algorithms import *

'''
Trains all policies for num_episodes (please set below).

If you would like to train some algorithms and not the other, please comment out the sections you do not want to train. This will not plot any data, but will store the Q dictionaries of state-action values in the Policies Folder.  
'''
num_episodes = 10000
b = BallBalancingTable(time_limit=60)

# '''
# Q-Learning Section: This will replace the existing policy. Please make sure to name it differently if you do not want to lose the work!
# '''
# steps, rewards, Q, win_perc = q_learning(b, num_episodes, 0.99, 0.01, 0.5)

# fin_Q = dict(Q)
# with open('Policies/q_learning_60s_3mil_L30.pickle', 'wb') as f:
#     pickle.dump(fin_Q, f)
    

# '''
# On-Policy Monte Carlo Control Section: This will replace the existing policy. Please make sure to name it differently if you do not want to lose the work!
# '''
# Q = on_policy_mc_control_epsilon_soft(b, num_episodes, 0.99, 0.1)

# # Save the resulting Q-values
# fin_Q = dict(Q)
# with open('Policies/mc_control_5s.pickle', 'wb') as f:
#     pickle.dump(fin_Q, f)

'''
Expected SARSA Section: This will replace the existing policy. Please make sure to name it differently if you do not want to lose the work!
'''
Q = exp_sarsa(b, num_episodes, 0.99, 0.1, 0.5)

# Save the resulting Q-values
fin_Q = dict(Q)
with open('Policies/exp_sarsa_5s.pickle', 'wb') as f:
    pickle.dump(fin_Q, f)



