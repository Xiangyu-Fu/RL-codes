import sys
import gym
import numpy as np

import matplotlib.collections as mc
import pandas as pd
import matplotlib.pyplot as plt
from K6_MountainCar_Agent import QLearningAgent


# 定义可用于离散空间的等间距网格。
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid


# 根据给定的网格离散样本。
def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # 返回索引值


# 在给定的二维网格上可视化原始的和离散的样本。
def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 显示网格
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)

    # 如果指定了边界(低、高)，则使用它们来设置轴的限制
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # 否则使用第一个、最后一个网格位置为low、high(为了进一步映射离散化的样本)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # 将每个离散的样本(实际上是一个索引)映射到相应网格单元格的中心
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples

    ax.plot(samples[:, 0], samples[:, 1], 'o')  # 绘制初始样本
    ax.plot(locs[:, 0], locs[:, 1], 's')  # 绘制离散后的样本
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)),
                                        colors='orange'))  # 添加一条线连接每个原始离散样本
    ax.legend(['original', 'discretized'])


def run(agent, env, num_episodes=20000, mode='train'):
    """给定的强化学习环境中运行agent并返回分数."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes + 1):
        # 初始化事件
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # 运行步骤直到完成
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        # 保存最终分数
        scores.append(total_reward)

        # 输出事件状态
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores


def plot_scores(scores, rolling_window=100):
    plt.plot(scores)
    plt.title("Scores")
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    return rolling_mean


def plot_q_table(q_table):
    """计算每个状态和相应动作的最大q值."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(q_image, cmap='jet')
    cbar = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')


# main function
if __name__ == "__main__":
    # 创建一个环境并设置随机种子
    env = gym.make("MountainCar-v0")
    env.seed(505)

    #  环境测试
    env_test = False
    if env_test is True:
        state = env.reset()
        score = 0
        for t in range(200):
            action = env.action_space.sample()
            env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        print("Final score:", score)
        env.close()

        # Explore state (observation) space
        print("State space:", env.observation_space)
        print("- low:", env.observation_space.low)
        print("- high:", env.observation_space.high)

        print("State space samples:")
        print(np.array([env.observation_space.sample() for i in range(10)]))

        # action space
        print("Action space:", env.action_space)

        # 从动作空间生成一些示例
        print("Action space samples:")
        print(np.array([env.action_space.sample() for i in range(10)]))

    # 将观测空间离散化,其中bins控制离散精度
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
    q_agent = QLearningAgent(env, state_grid)

    # 以不同模式运行,方便测试结果
    run_mode = True
    # 运行测试模式
    if run_mode is True:
        q_agent.q_table = np.load('q_table.npy', allow_pickle=True)
        state = env.reset()
        score = 0
        for t in range(200):
            action = q_agent.act(state, mode='test')
            env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        print('Final score:', score)
        env.close()
    # 训练模式
    else:
        scores = run(q_agent, env, num_episodes=50000)

        # plot data
        plt.plot(scores)
        plt.title("Scores")
        rolling_mean = plot_scores(scores)
        plt.show()

        test_scores = run(q_agent, env, num_episodes=100, mode='test')
        print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
        _ = plot_scores(test_scores)

        plot_q_table(q_agent.q_table)
        plt.show()
        np.save('q_table.npy', q_agent.q_table)
