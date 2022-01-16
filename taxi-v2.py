from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=20000)
state = env.reset()
while True:
    action = np.argmax(agent.Q[state]) if state is not None else 0
    print("Q[state] = {}\naction = {}".format(agent.Q[state], action))
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
