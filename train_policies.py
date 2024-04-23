from env import BallBalancingTable
import matplotlib.pyplot as plt
from env import  *
from policy import *
import matplotlib.pyplot as plt
import pickle
from myalgs import *

'''
Trains all policies for num_episodes (please set below).

If you would like to train some algorithms and not the other, please comment out the sections you do not want to train. This will not plot any data, but will store the Q dictionaries of state-action values in the Policies Folder.  
'''
num_episodes = 3000000
b = BallBalancingTable(time_limit=60)

'''
Q-Learning Section: This will replace the existing policy. Please make sure to name it differently if you do not want to lose the work!
'''
steps, rewards, Q, win_perc = q_learning(b, num_episodes, 0.99, 0.01, 0.5)

fin_Q = dict(Q)
with open('Policies/q_learning_60s_3mil_L30.pickle', 'wb') as f:
    pickle.dump(fin_Q, f)



