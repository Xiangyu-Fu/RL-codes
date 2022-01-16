# Import common libraries
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

'''
(1) 可视化函数
'''


# 以网格形式可视化每个瓦片
def visualize_tilings(tilings):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']
    legend_lines = []

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        for x in grid[0]:
            l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
        for y in grid[1]:
            l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        legend_lines.append(l)
    ax.grid('off')
    ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
    ax.set_title("Tilings")
    return ax


def visualize_encoded_samples(samples, encoded_samples, tilings, low=None, high=None):
    """Visualize samples by activating the respective tiles."""
    samples = np.array(samples)  # for ease of indexing

    # Show tiling grids
    ax = visualize_tilings(tilings)

    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Pre-render (invisible) samples to automatically set reasonable axis limits, and use them as (low, high)
        ax.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.0)
        low = [ax.get_xlim()[0], ax.get_ylim()[0]]
        high = [ax.get_xlim()[1], ax.get_ylim()[1]]

    # Map each encoded sample (which is really a list of indices) to the corresponding tiles it belongs to
    tilings_extended = [np.hstack((np.array([low]).T, grid, np.array([high]).T)) for grid in
                        tilings]  # add low and high ends
    tile_centers = [(grid_extended[:, 1:] + grid_extended[:, :-1]) / 2 for grid_extended in
                    tilings_extended]  # compute center of each tile
    tile_toplefts = [grid_extended[:, :-1] for grid_extended in tilings_extended]  # compute topleft of each tile
    tile_bottomrights = [grid_extended[:, 1:] for grid_extended in tilings_extended]  # compute bottomright of each tile

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for sample, encoded_sample in zip(samples, encoded_samples):
        for i, tile in enumerate(encoded_sample):
            # Shade the entire tile with a rectangle
            topleft = tile_toplefts[i][0][tile[0]], tile_toplefts[i][1][tile[1]]
            bottomright = tile_bottomrights[i][0][tile[0]], tile_bottomrights[i][1][tile[1]]
            ax.add_patch(Rectangle(topleft, bottomright[0] - topleft[0], bottomright[1] - topleft[1],
                                   color=colors[i], alpha=0.33))

            # In case sample is outside tile bounds, it may not have been highlighted properly
            if any(sample < topleft) or any(sample > bottomright):
                # So plot a point in the center of the tile and draw a connecting line
                cx, cy = tile_centers[i][0][tile[0]], tile_centers[i][1][tile[1]]
                ax.add_line(Line2D([sample[0], cx], [sample[1], cy], color=colors[i]))
                ax.plot(cx, cy, 's', color=colors[i])

    # Finally, plot original samples
    ax.plot(samples[:, 0], samples[:, 1], 'o', color='r')

    ax.margins(x=0, y=0)  # remove unnecessary margins
    ax.set_title("Tile-encoded samples")
    return ax


def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores)
    plt.title("Scores")
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    return rolling_mean


'''
(2) Tiling
'''


# 创建瓦片化网格
def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    return [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]


# 瓦片化
def create_tilings(low, high, tiling_specs):
    return [create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs]


'''
(3) Tile Encoding
'''


# 根据给定的网格离散样本。
def discretize(sample, grid):
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # 返回索引值


# 使用瓦片编码对给定的样本进行编码
def tile_encode(sample, tilings, flatten=False):
    encoded_sample = [discretize(sample, grid) for grid in tilings]  # 返回在相应瓦片上的坐标
    return np.concatenate(encoded_sample) if flatten else encoded_sample


'''
(4) 使用瓦片编码的Q表
'''


class QTable:
    # 初始化Q表
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Create Q-table, initialize all Q-values to zero
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("QTable(): size =", self.q_table.shape)


class TiledQTable:
    """组合q表与其内部的瓦片编码"""
    # 瓦片化并初始化内部q表。
    def __init__(self, low, high, tiling_specs, action_size):
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits)+1 for splits in tiling_grid)
                            for tiling_grid in self.tilings]    # 每片瓦片上的状态数
        self.action_size = action_size
        self.q_tables = [QTable(state_size, self.action_size)
                         for state_size in self.state_sizes]    # 为每片瓦片建立一个q表
        print("TiledQTable(): no. of internal tables = ", len(self.q_tables))

    # 得到给定<状态，动作>对的q值。
    def get(self, state, action):
        # 获取在瓦片上的位置索引
        encoded_state = tile_encode(state, self.tilings)
        # 检索每片瓦片上的q值，并返回它们的平均值
        value = 0.0
        for idx, q_table in zip(encoded_state, self.q_tables):  # 在每片瓦片上循环
            value += q_table.q_table[tuple(idx + (action,))]
        value = value / len(self.q_tables)                      # 计算平均值
        return value

    # 软更新q值为给定<状态，行动>对的值
    def update(self, state, action, value, alpha=0.1):
        # 获取在瓦片上的位置索引
        encoded_state = tile_encode(state, self.tilings)
        # 通过学习率alpha更新每个瓦片上的的q值
        for idx, q_table in zip(encoded_state, self.q_tables):
            value_ = q_table.q_table[tuple(idx + (action,))]   # 获取当前位置q表的值
            q_table.q_table[tuple(idx + (action,))] = alpha * value + (1.0 - alpha) * value_        # 使用学习率更新相关位置的值


'''
(5) Q学习智能体
'''


class QLearningAgent:
    def __init__(self, env, tiled_q_table, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """初始化变量，创建离散化网格。"""
        # Environment info
        self.env = env
        self.tq = tiled_q_table
        self.state_sizes = tiled_q_table.state_sizes
        self.action_size = self.env.action_space.n  # 1-维离散动作空间
        self.seed = np.random.seed(seed)
        print("--Agent--\nEnvironment:", self.env)
        print("State space size:", self.state_sizes)
        print("Action space size:", self.action_size)

        # 学习模型参数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = self.initial_epsilon = epsilon  # 初始探索率
        self.epsilon_decay_rate = epsilon_decay_rate  # epsilon衰减系数
        self.min_epsilon = min_epsilon

    def reset_episode(self, state):
        """为新的事件重置变量."""
        # 逐步降低探索率
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # 决定初始行动
        self.last_state = state
        Q_s = [self.tq.get(state, action) for action in range(self.action_size)]
        self.last_action = np.argmax(Q_s)
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """重置训练时使用的探索率."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """选择next操作并更新内部Q表 (when mode != 'test')."""
        Q_s = [self.tq.get(state, action) for action in range(self.action_size)]
        greedy_action = np.argmax(Q_s)
        if mode == 'test':
            # 测试模式:简单地产生一个动作
            action = greedy_action
        else:
            # 训练模式(默认):更新Q表，选择下一步行动
            # Note: 我们用当前状态,回报更新最后的状态动作对的Q表条目
            value = reward + self.gamma * max(Q_s)
            self.tq.update(self.last_state, self.last_action, value, self.alpha)

            # 探索 vs. 利用
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # 随机选择一个动作
                action = np.random.randint(0, self.action_size)
            else:
                # 从Q表中选择最佳动作
                action = greedy_action

        # 存储当前状态，下一步操作
        self.last_state = state
        self.last_action = action
        return action


'''
(6) 模型训练
'''
def run(agent, env, num_episodes=10000, mode='train'):
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # 初始化环境
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        #  保存最终成绩
        scores.append(total_reward)

        if mode == "train":
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 10 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()
    return scores


if __name__ == "__main__":
    # Set plotting options
    plt.style.use('ggplot')
    np.set_printoptions(precision=3, linewidth=120)

    # Create an environment
    env = gym.make('Acrobot-v1')
    env.seed(505)

    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    test = create_tiling_grid(low, high, bins=(10, 10), offsets=(-0.1, 0.5))

    # 设置分割精度
    n_bins = 5
    bins = tuple([n_bins] * env.observation_space.shape[0])
    offset_pos = (env.observation_space.high - env.observation_space.low) / (3 * n_bins)

    tiling_specs = [(bins, -offset_pos),
                    (bins, tuple([0.0] * env.observation_space.shape[0])),
                    (bins, offset_pos)]

    tq = TiledQTable(env.observation_space.low,
                     env.observation_space.high,
                     tiling_specs,
                     env.action_space.n)
    agent = QLearningAgent(env, tq)

    scores = run(agent, env)

    rolling_mean = plot_scores(scores)

    '''
    # Tiling specs: [(<bins>, <offsets>), ...]
    tiling_specs = [((10, 10), (-0.066, -0.33)),
                    ((10, 10), (0.0, 0.0)),
                    ((10, 10), (0.066, 0.33))]
    tilings = create_tilings(low, high, tiling_specs)
    # visualize_tilings(tilings)
    '''

    '''
    # Test with some sample values
    samples = [(-1.2, -5.1),
               (-0.75, 3.25),
               (-0.5, 0.0),
               (0.25, -1.9),
               (0.15, -1.75),
               (0.75, 2.5),
               (0.7, -3.7),
               (1.0, 5.0)]
    encoded_samples = [tile_encode(sample, tilings) for sample in samples]
    print("\nSamples:", repr(samples), sep="\n")
    print("\nEncoded samples:", repr(encoded_samples), sep="\n")
    # visualize_encoded_samples(samples, encoded_samples, tilings)
    plt.show()
    '''

    '''
    # Test with a sample Q-table
    tq = TiledQTable(low, high, tiling_specs, 2)
    s1 = 3;
    s2 = 4;
    a = 0;
    q = 1.0
    print("[GET]    Q({}, {}) = {}".format(samples[s1], a,
                                           tq.get(samples[s1], a)))  # check value at sample = s1, action = a
    print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q));
    tq.update(samples[s2], a, q)  # update value for sample with some common tile(s)
    print("[GET]    Q({}, {}) = {}".format(samples[s1], a,
                                           tq.get(samples[s1], a)))  # check value again, should be slightly updated
    '''
