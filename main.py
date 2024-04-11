from env import *
from algorithms import *
import matplotlib.pyplot as plt

def plot(env, num_steps, gamma=1, epsilon=0.1, step_size=0.5):
    # SARSA
    episode_count,tag=sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_ep - std_error1,avg_ep + std_error1,alpha=0.4)

    # EXP_SARSA
    episode_count,tag=exp_sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0, num_steps, 1), avg_ep - std_error1,avg_ep + std_error1,alpha=0.4)

    # N-Step Sarsa
    episode_count,tag=nstep_sarsa(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0, num_steps, 1), avg_ep - std_error1, avg_ep + std_error1, alpha=0.4)

    #MC on Policy
    episode_count,tag=on_policy_mc_control_epsilon_soft(env,num_steps,gamma,epsilon)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0, num_steps, 1), avg_ep - std_error1, avg_ep + std_error1, alpha=0.4)

    #Q-Learning
    episode_count,tag=q_learning(env,num_steps,gamma,epsilon,step_size)
    avg_ep = np.mean(np.array(episode_count),0)
    plt.plot(np.arange(0,num_steps,1),avg_ep,label=tag)
    std_error1 = 1.96*np.std(episode_count)/np.sqrt(10)
    plt.fill_between(np.arange(0,num_steps,1),avg_ep - std_error1,avg_ep + std_error1,alpha=0.4)

    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("episodes")
    plt.show()

if __name__ == "__main__":
    env = BallBalancingTable(sensor_noise=False, sensor_std= 0.005, sensor_sensitivity=2, ball_mass=0.5, table_mass=10, table_length=0.3, dt=0.1, force_step=0.5, angle_limit=30, force_limit=10, max_damping=3)
    plot(env, num_steps=8000)
