import sys
import gym
import numpy as np
from collections import defaultdict  # default函数返回一个默认值

from plot_utils import plot_blackjack_values, plot_policy


# Part1: MC Prediction
def generate_spisode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)  # 以prods的概率分布选择动作
        next_state, reward, done, info = bj_env.step(action)
        # 返回状态, 动作, 回报
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries fo arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))  # lambda定义了一个匿名函数
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # 过程跟踪
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()  # 使缓存区显示输出
        # 生成一个事件
        episode = generate_episode(env)
        # 获得状态、行动和奖励
        states, actions, rewards = zip(*episode)  # 解压episode
        # 返回折扣率
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        # 更新事件中返回的总和、访问次数和对所有状态-动作对的动作-价值函数估计
        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
    return Q


# Part2: MC Control
def get_probs(Q_s, epsilon, nA):
    """ 得到与epsilon-贪婪策略对应的动作概率"""
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon/nA)
    return policy_s


def update_Q(env, episode, Q, alpha, gamma):
    """ 使用最近的事件更新动作-价值函数评估"""
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q


def generate_episode_from_Q(env, Q, epsilon, nA):
    """ 根据执行epsilon-贪婪策略生成一个事件"""
    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.array(nA), p=get_probs(Q[state], epsilon, nA)) \
            if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.9999, eps_min=0.05):
    # nA = 动作数
    nA = env.action_space.n
    # 初始化数组的空字典
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    for i_episode in range(1, num_episodes+1):
        # 监控进程
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        # 根据epsilon-贪婪政策生成一个事件
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k, np.argmax(v))for k, v in Q.items())

    return policy, Q


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')

    print(env.observation_space)
    print(env.action_space)

    '''Part1: MC Prediction'''
    # 获得动作价值函数
    Q = mc_prediction_q(env, 50000, generate_spisode_from_limit_stochastic)

    # 获得相应的状态价值函数
    V_to_plot = dict((k, (k[0] > 18)*(np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v)))
                     for k, v in Q.items())

    # 状态价值函数图
    plot_blackjack_values(V_to_plot)

    '''Part2: MC Control'''
    # 获得估计的最优策略和行动-价值函数
    policy, Q = mc_control(env, 50000, 0.02)

    # 得到相应的状态值函数
    V = dict((k, np.max(v)) for k, v in Q.items())

    # 绘制状态值函数
    plot_blackjack_values(V)

    # 绘制相应的策略
    plot_policy(policy)
