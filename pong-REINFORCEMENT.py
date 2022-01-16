import pong_utils
import gym
import time

import matplotlib
import matplotlib.pyplot as plt
device = pong_utils.device
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1 - P(right)
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(16, 64, kernel_size=2, stride=2)

        self.size = 32 * 5 * 5

        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


def discounted_future_rewards(rewards, ratio=0.999):
    n = rewards.shape[1]
    step = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    ones = torch.ones_like(step)
    zeros = torch.zeros_like(step)

    target = torch.where(step >= 0, ones, zeros)
    step = torch.where(step >= 0, step, zeros)
    discount = target * (ratio ** step)
    discount = discount.to(device)

    rewards_discounted = torch.mm(rewards, discount)
    return rewards_discounted


def surrogate(policy, old_probs, states, actions, rewards, discount=0.995, beta=0.01):
    actions = torch.tensor(actions, dtype=torch.int8)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)

    # convert states to policy(or probability)
    new_probs =


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env = gym.make('PongDeterministic-v4')
    print('using device :',device)
    print("List of available actions", env.unwrapped.get_action_meanings())

    env.reset()
    _, _, _, _ = env.step(0)
    for _ in range(20):
        frame, _, _, _ = env.step(1)

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('original image')

    plt.subplot(1, 2, 2)
    plt.title('preprocessed image')

    # 80 x 80 black and white image
    plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
    plt.show()

    policy = pong_utils.Policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    pong_utils.play(env, policy, time=100)


if __name__ == '__main__':
    main()