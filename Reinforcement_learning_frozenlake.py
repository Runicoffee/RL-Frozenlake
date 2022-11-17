import gym
import numpy as np
import matplotlib.pyplot as plt

import random
#make frozenlakes slippery and non slippery
def make_frozenlake():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.reset()
    return env

def make_slippery_frozenlake():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env.reset()
    return env

#random action 0-3
def random_gererator():
    r = random.randint(0,3)
    return(r)
#Q elarning.
def Qlearning(episodes, max_steps, deterministic, env):
    #0/left 1/down 2/right 3/up
    #initialize Qtable
    Q = np.zeros([16,4])
    learning_rate = 0.9
    alpha = 0.5
    #down, down, righ, down, right right
    for i in range(episodes):
        done = False
        curr_state = env.reset()
        for j in range(max_steps):
            action = random_gererator()
            new_state, reward, done, info = env.step(action)
            #deterministic uppdate rule or not.
            if deterministic == True:
                Q[curr_state,action] = reward + learning_rate*np.max(Q[new_state,:])
            else:
                Q[curr_state,action] = Q[curr_state,action] + alpha*(reward + learning_rate*np.max(Q[new_state,:]) - Q[curr_state,action])
            curr_state=new_state
            if done == True:#when in hole or in goal, end this episode.
                break
    return Q
#evaluation function, basically the same as in lecture notes.
def evaluation(Q, test_episodes, steps, env):
    rewards = []
    for episode in range(test_episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        for step in range(steps):
            action = np.argmax(Q[state,:])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
            #env.render()
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    avg_reward = sum(rewards)/test_episodes
    return avg_reward

#this script runs the learnings and plots the lines for all 3 cases.
def frozenlake_learn_and_plot_script(deterministic,slippery,graph_id):
    total_episodes = 300
    max_steps = 100
    for i in range(10):#every case 10 times so we get ten lines.
        if slippery:
            env = make_slippery_frozenlake()
        else:
            env = make_frozenlake()
        avg_rw = []
        x = []
        episodes = 10
        #learning and evaluation and ploting.
        while episodes <= total_episodes:
            Q = Qlearning(episodes, max_steps, deterministic, env)
            #100 max steps and 50 test episodes should be good enough.
            avg_rw.append(evaluation(Q,50,100,env))
            x.append(episodes)
            episodes += 10
        axs[graph_id].plot(x, avg_rw)
        env.close()

fig, axs = plt.subplots(3)#plot the initial page
print("learning, testing and plotting...")
#A) deterministic case with determinsicti 
frozenlake_learn_and_plot_script(True, False, 0)
#B) non deterministic with deterministic update rule
frozenlake_learn_and_plot_script(True, True, 1)
#C) non deterministic with non deterministi uppdata rule
frozenlake_learn_and_plot_script(False, True, 2)
#add titles
axs[0].set_title('Deterministic update rule & Not slippery')
axs[1].set_title('Deterministic update rule & Slippery')
axs[2].set_title('Non-Deterministic update rule & Slippery')

fig.canvas.draw()
plt.show()
