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
num_episodes = 1000000
b = BallBalancingTable()

'''
Q-Learning Section: This will replace the existing policy. Please make sure to name it differently if you do not want to lose the work!
'''
steps, rewards, Q, win_perc = q_learning(b, num_episodes, 0.99, 0.01, 0.5)

fin_Q = dict(Q)
with open('Policies/q_learning.pickle', 'wb') as f:
    pickle.dump(fin_Q, f)



