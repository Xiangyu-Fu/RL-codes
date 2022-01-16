import gym
env = gym.make('CliffWalking-v0')
print('观察空间 = {}'.format(env.observation_space))
print('= {}'.format(env.action_space))
print(' = {}, = {}'.format(env.nS, env.nA))
print(' = {}'.format(env.space))


import numpy as np
def play_once(env, policy):
    total_reward = 0
    state = env.reset()
    loc = np.unravel_index(state, env.shape)
    print('状态 = {} ， 位置={} '.format(state, loc))
    while True:
        action = np.random.choice(env.nA, p=policy[state])
        next_state, reward, done, _ = env.step(action)
        print('状态 = {}, 位置 ={}, 奖励 ={}'.format(state, loc, reward))
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward


actions = np.ones(env.shape, dtype=int)
actions[-1, :] = 0
actions[:, -1] = 2
optimal_policy = np.eye(4)[actions.reshape(-1)]


def evaluate_bellman(env, policy, gamma=1.):
    a, b = np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for p, next_state, reward, done in env.P[state][action]:
                a[state, next_state] -= (pi * gamma)
                b[state] += (pi * reward * p)
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.P[state][action]:
                q[state][action] += ((reward + gamma * v[next_state]) * p)
    return v, q


policy = np.random.uniform(size=(env.nS, env.nA))
policy = policy / np.sum(policy, axis=1)[:, np.newaxis]
state_values, action_values = evaluate_bellman(env, policy)
print('状态价值 ={}'.format(state_values))
print('动作价值 ={}'.format(action_values))


optimal_state_values, optimal_action_values = evaluate_bellman(env, optimal_policy)
print('最优状态价值 = {}'.format(optimal_state_values))
print('最优动作价值 = {}'.format(optimal_action_values))


import scipy


def optimal_be1lman(env, gamma=1.):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[state][action]:
                p[state, action, next_state] += prob
                r[state, action] += (reward * prob)
    c = np.ones(env.nS)
    a_ub = gamma * p.reshape(-1, env.nS) - np.repeat(np.eye(env.nS), env.nA, axis=0)
    b_ub = -r.reshape(-1)
    a_eq = np.zeros((0, env.nS))
    b_eq = np.zeros(0)
    bounds = [(None, None), ] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds, method='interior-point')
    v = res.x
    q = r + gamma * np.dot(p, v)
    return v, q


optimal_state_values, optimal_action_values = optimal_be1lman(env)
print('最优状态价值 ={}'.format(optimal_state_values))
print('最优动作价值 ={}'.format(optimal_action_values))


optimal_actions = optimal_action_values.argmax(axis=1)
print('最优策略 ={}'.format(optimal_actions))