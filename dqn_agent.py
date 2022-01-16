import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # 经验回放的缓冲区的大小
BATCH_SIZE = 64  # 最小训练批数量
GAMMA = 0.99  # 折扣率
TAU = 1e-3  # 用于目标函数的柔性策略更新
LR = 5e-4  # 学习率
UPDATE_EVERY = 4  # 更新网络的频率

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    """与环境相互作用，从环境中学习。"""

    def __init__(self, state_size, action_size, seed, debug_mode=False):
        """初始化智能体对象。
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.debug_mode = debug_mode
        print('Program running in {}'.format(device))

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)  # 自适应梯度算法
        # print('Q-Network_local:{}\nQ-Network_target:{}'.format(self.qnetwork_local, self.qnetwork_target))

        # 经验回放
        if self.debug_mode is True:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, 1, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # 初始化时间步 (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # 在经验回放中保存经验
        self.memory.add(state, action, reward, next_state, done)

        # 在每个时间步UPDATE_EVERY中学习
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # 如果内存中有足够的样本，取随机子集进行学习
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        if self.debug_mode is True:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """根据当前策略返回给定状态的操作.
        
        Params
        ======
            state (array_like): 当前的状态
            eps (float): epsilon, 用于 epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # 将qn更改成评估形式
        self.qnetwork_local.eval()
        # 禁用梯度
        with torch.no_grad():
            # 获得动作价值
            action_values = self.qnetwork_local(state)
        # 将qn更改成训练模式
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """使用给定的一批经验元组更新值参数。

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        if self.debug_mode is True:
            print('\nstates={}, actions={}, rewards={}, next_states={}, dones={}'
              .format(states, actions, rewards, next_states, dones))

        # compute and minimize the loss
        # 从目标网络得到最大的预测Q值(下一个状态)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # 计算当前状态的Q目标
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # 从评估网络中获得期望的Q值
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        if self.debug_mode is True:
            print('Q_target_next={}, \nQ_target ={}, \nQ_expected={},'
                  .format(Q_targets_next, Q_targets, Q_expected))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # 执行单个优化步骤
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """:柔性更新模型参数。
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): 插值参数
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # 柔性更新, 将src中数据复制到self中
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """在memory中添加一段新的经验."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """从memory中随机抽取一批经验."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
