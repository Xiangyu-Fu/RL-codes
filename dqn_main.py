import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import os


def dqn(n_episode=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, mode='train'):
    """
    Deep Q-Learning

    :param n_episode:maximum number of training episodes
    :param max_t:maximum number of timesteps per episode
    :param eps_start:starting value of epsilon, for epsilon-greedy action selection
    :param eps_end:minimum value of epsilon
    :param eps_decay:multiplicative factor (per episode) for decreasing epsilon
    :return: final score
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    if mode == 'train':
        for i_episode in range(1, n_episode+1):
            # 初始化状态
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay*eps)
            print('\rEpisode {}\t score:{} \tAverage Score:{:.2f}'.format(i_episode, score,np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\rAverage Score :{:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episode! \t Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
    else:
        # 训练一次
        state = env.reset()
        for j in range(200):
            action = agent.act(state, eps)
            print('state :{} action :{}'. format(state, action))
            env.render()
            next_state, reward, done, _ = env.step(action)
            print('next_state={}, reward={}, done={}'.format(next_state, reward, done))
            agent.step(state, action, reward, next_state, done)
            if done:
                break

    return scores


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    MODE = 'train'

    if MODE == 'debug':
        # 调试模式
        agent = Agent(state_size=8, action_size=4, seed=1,debug_mode=True)
        scores = dqn(mode='test')

    elif MODE == 'run':
        agent = Agent(state_size=8, action_size=4, seed=1)
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

        # 以当前策略运行
        for i in range(3):
            state = env.reset()
            for j in range(200):
                action = agent.act(state)
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break

        env.close()
    else:
        # 训练模式
        agent = Agent(state_size=8, action_size=4, seed=1)
        scores = dqn()

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
