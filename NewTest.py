import gym
import numpy as np
import scipy


# 单回合运行
def play_once(env, policy):
    total_reward = 0
    state = env.reset()
    while True:
        action = np.random.choice(env.nA, p=policy[state])
        next_state, reward, done, _ = env.step(action)
        # np.unravel_index函数的作用是获取一个/组int类型的索引值在一个多维数组中的位置。
        loc = np.unravel_index(state, env.shape)
        print('状态 = {}, 位置 ={}, 奖励 ={}'.format(state, loc, reward))
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward


# 用Bellman方程求解状态价值和动作价值
def evaluate_bellman(env, policy, gamma=1.):
    a, b = np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            # print('{}{}:pi:{}'.format(state, action, pi))
            # env.P中存储环境的动力
            for p, next_state, reward, done in env.P[state][action]:
                a[state, next_state] -= (pi * gamma)
                b[state] += (pi * reward * p)
    # np.linalg.solve()函数求解标准形式的线性方程组
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    # 利用状态价值函数求解动作价值函数
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.P[state][action]:
                q[state][action] += ((reward + gamma * v[next_state]) * p)
    return v, q


# 使用动态规划求解最优策略
def optimal_bellman(env, gamma=1.):
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


# 主函数
if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('状态数量 = {}, 动作数量= {}'.format(env.nS, env.nA))
    print('地图大小 = {}'.format(env.shape))

    ''' 
    # 定义最优策略
    actions = np.ones(env.shape, dtype=int)
    actions[-1, :] = 0
    actions[:, -1] = 2
    optimal_policy = np.eye(4)[actions.reshape(-1)]
    

    total_reward = play_once(env, optimal_policy)
    print('总奖励 ={}'.format(total_reward))
    
    optimal_state_values, optimal_action_values = evaluate_bellman(env, optimal_policy)
    print('最优状态价值 = {}'.format(optimal_state_values.reshape(4, -1)))
    print('最优动作价值 = {}'.format(optimal_action_values))
    '''

    '''
    # 定义随机策略
    policy = np.random.uniform(size=(env.nS, env.nA))
    # 归一化
    policy = policy / np.sum(policy, axis=1)[:, np.newaxis]
    state_values, action_values = evaluate_bellman(env, policy)
    print('状态价值 ={}'.format(state_values))
    print('动作价值 ={}'.format(action_values))
    '''

    # 求解最优策略
    optimal_state_values, optimal_action_values = optimal_bellman(env)
    print('最优状态价值 ={}'.format(optimal_state_values))
    print('最优动作价值 ={}'.format(optimal_action_values))

    optimal_actions = optimal_action_values.argmax(axis=1)
    print('最优策略 ={}'.format(optimal_actions))
