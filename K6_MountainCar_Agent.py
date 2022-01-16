import numpy as np


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


class QLearningAgent:
    """Q-Learning agent，，通过离散化可以作用于连续的状态空间。"""

    def __init__(self, env, state_grid, alpha=0.05, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """初始化变量，创建离散化网格。"""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-维状态空间
        self.action_size = self.env.action_space.n  # 1-维离散动作空间
        self.seed = np.random.seed(seed)
        print("--Agent--\nEnvironment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)

        # 学习模型参数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = self.initial_epsilon = epsilon  # 初始探索率
        self.epsilon_decay_rate = epsilon_decay_rate  # epsilon衰减系数
        self.min_epsilon = min_epsilon

        # 创建Q表
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """将连续状态映射到它的离散表示。"""
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """为新的事件重置变量."""
        # 逐步降低探索率
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # 决定初始行动
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """重置训练时使用的探索率."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """选择next操作并更新内部Q表 (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == 'test':
            # 测试模式:简单地产生一个动作
            action = np.argmax(self.q_table[state])
        else:
            # 训练模式(默认):更新Q表，选择下一步行动
            # Note: 我们用当前状态,回报更新最后的状态动作对的Q表条目
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                                                                   (reward + self.gamma * max(self.q_table[state]) -
                                                                    self.q_table[self.last_state + (self.last_action,)])

            # 探索 vs. 利用
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # 随机选择一个动作
                action = np.random.randint(0, self.action_size)
            else:
                # 从Q表中选择最佳动作
                action = np.argmax(self.q_table[state])

        # 存储当前状态，下一步操作
        self.last_state = state
        self.last_action = action
        return action