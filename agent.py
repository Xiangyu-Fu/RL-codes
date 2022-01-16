import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.003    # epsilon值
        self.gamma = 0.99    # 折扣率
        self.alpha = 1    # 学习率

    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            next_action = self.select_action(state)
            current = self.Q[state][action]

            policy_s = np.ones(self.nA) * self.eps / self.nA
            policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
            Qsa_next = np.dot(self.Q[next_state], policy_s)

            target = reward + (self.gamma * Qsa_next)
            self.Q[state][action] = current + (target - current)
            '''
            Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0  # value of next state
            target = reward + (self.gamma * Qsa_next)  # construct TD target
            self.Q[state][action] = current + (self.alpha * (target - current))  # get updated value
            '''
