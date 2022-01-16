import sys
import gym
import numpy as np
import random
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt

from plot_utils_TD import plot_values


# Part 1: TD Control: Sarsa
def update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    """
    返回根据当前位置的最新的Q表值
    :param alpha: 学习率
    :param gamma: 折扣率
    :param Q: Q表
    :param state: 当前状态
    :param action: 当前动作
    :param reward: 回报
    :param next_state: 下个时间步的状态
    :param next_action: 下个时间步的动作
    :return: 新的Q值
    """
    current = Q[state][action]  # 在Q表中的预测 (对于当前的状态-动作对)
    # 得到下一个时间步的状态-动作对的值
    # 详情参考公式(5.7)
    Qsa_next = Q[next_state][next_action] if next_state is not None else 0
    target = reward + (gamma * Qsa_next)  # 构建TD目标
    new_value = current + (alpha * (target - current))
    return new_value


def epsilon_greedy(Q, state, nA, eps):
    """
    为提供的状态选择epsilon-贪婪动作
    :param Q: (dictionary)动作价值函数
    :param state: (int)当前状态
    :param nA: (int)环境中的动作数
    :param eps: epsilon
    """
    if random.random() > eps:  # 选择概率为epsilon的贪婪动作
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(env.action_space.n))


def sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    nA = env.action_space.n  # 动作的数量
    Q = defaultdict(lambda: np.zeros(nA))  # 初始化数组的空字典

    # 性能监控
    tmp_scores = deque(maxlen=plot_every)  # 建立关于分数双向队列对象
    avg_scores = deque(maxlen=num_episodes)  # 建立平均分数的双向队列对象
    for i_episode in range(1, num_episodes+1):
        # 过程跟踪
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        score = 0  # 初始化分数
        state = env.reset()  # 开始一个事件

        eps = 1.0 / i_episode   # 设置epsilon的值
        action = epsilon_greedy(Q, state, nA, eps)   # 采取 epsilon-贪婪动作

        while True:
            next_state, reward, done, info = env.step(action)   # take action A, observe R, S'
            score += reward  # 给智能体的分数中添加奖励
            if not done:
                next_action = epsilon_greedy(Q, next_state, nA, eps)   # 采取 epsilon-贪婪 动作
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q,
                                                  state, action, reward, next_state, next_action)  # 更新Q表

                state = next_state  # S <- S'
                action = next_action  # A <- A'

            if done:
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, state, action, reward)
                tmp_scores.append(score)  # 添加分数
                break
        # 更新平均分数
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('\nBest Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return Q


# Part2: TD Control: Q-学习
def update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state=None):
    """ Return updated Q-value for the most recent experience """
    current = Q[state][action]  # estimate in Q-table (for current state, action pair)
    Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # 选择最大状态价值
    target = reward + (gamma * Qsa_next)  # 构建 TD 目标
    new_value = current + (alpha * (target - current))  # get updated value
    return new_value


def q_learning(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    """
    Q-Learning - TD Control
    :param num_episodes: (int) number of episodes to run the algorithm
    :param alpha: (float)学习率
    :param gamma: (float)折扣率
    :param plot_every: (int) 在计算平均分数时使用的集数
    """
    nA = env.action_space.n  # 动作数
    Q = defaultdict(lambda: np.zeros(nA))  # 初始化数组的空字典

    # monitor performance
    tmp_scores = deque(maxlen=plot_every)  # 记录分数的deque
    avg_scores = deque(maxlen=num_episodes)  # 每固定集的平均得分

    for i_episode in range(1, num_episodes+1):
        # monitor process
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end='')
            sys.stdout.flush()
        score = 0
        state = env.reset()
        eps = 1.0 / i_episode

        while True:
            action = epsilon_greedy(Q, state, nA, eps)
            next_state, reward, done, info = env.step(action)
            score += reward
            Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state)

            state = next_state
            if done:
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return Q


# Part 3: TD Control: Expected Sarsa
def update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state=None):
    """
    返回根据当前位置的最新的Q表值
    :param alpha: 学习率
    :param gamma: 折扣率
    :param nA: 动作数
    :param eps: epsilon
    :param Q: Q表
    :param state: 当前状态
    :param action: 当前动作
    :param reward: 回报
    :param next_state:下个状态
    """
    current = Q[state][action]         # estimate in Q-table (for current state, action pair)

    policy_s = np.ones(nA) * eps / nA  # 当前的政策(for next state S')
    policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA)  # 贪婪动作
    Qsa_next = np.dot(Q[next_state], policy_s)          # 获取下一时刻状态的值

    target = reward + (gamma * Qsa_next)                # 目标构建
    new_value = current + (alpha * (target - current))  # 获得更新后的值
    return new_value


def expected_sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    """Expected SARSA - TD Control

    Params
    ======
        num_episodes (int): number of episodes to run the algorithm
        alpha (float): step-size parameters for the update step
        gamma (float): discount factor
        plot_every (int): number of episodes to use when calculating average score
    """
    nA = env.action_space.n  # number of actions
    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays

    # monitor performance
    tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)  # average scores over every plot_every episodes

    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        score = 0  # initialize score
        state = env.reset()  # start episode
        eps = 0.005  # set value of epsilon

        while True:
            action = epsilon_greedy(Q, state, nA, eps)  # epsilon-greedy action selection
            next_state, reward, done, info = env.step(action)  # take action A, observe R, S'
            score += reward  # add reward to agent's score
            # update Q
            Q[state][action] = update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state)
            state = next_state  # S <- S'
            if done:
                tmp_scores.append(score)  # append score
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return Q


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    print("Action space:{}".format(env.action_space.n))
    print("Observation space : {}".format(env.observation_space.n))

    '''Part 1: TD Control: Sarsa'''
    # 得到估计的最优策略和相应的行动-价值函数
    Q_sarsa = sarsa(env, 5000, .01)
    print(Q_sarsa.items())

    # 输出估计的最优策略
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4, 12)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # 绘制估计的最优状态价值函数
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)

    '''Part 2: TD Control: Q-learning'''
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsamax = np.array(
        [np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4, 12))

    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])

    '''Part 3: TD Control: Expected Sarsa'''
    # obtain the estimated optimal policy and corresponding action-value function
    Q_expsarsa = expected_sarsa(env, 5000, 1)

    # print the estimated optimal policy
    policy_expsarsa = np.array(
        [np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4, 12)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
