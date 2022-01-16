import gym
import numpy as np


# 使用策略执行一个回合
def play_policy(env, policy, render=False):
    total_reward = 0.
    observation = env.reset()
    while True:
        if render:
            env.render()  # 此行可显示
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward   # 统计回合奖励
        if done:  # 游戏结束
            break
    return total_reward


# 策略评估的实现
def v2q(env, v, s=None, gamma=1.):
    if s is not None:
        q = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q[a] += prob * (reward + gamma * v[next_state] * (1. - done))
    else:
        q = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            q[s] = v2q(env, v, s, gamma)
    return q


# 定义策略评估函数
def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v = np.zeros(env.unwrapped.nS)
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vs = sum(policy[s] * v2q(env, v, s, gamma))
            delta = max(delta, abs(v[s]-vs))
            v[s] = vs
        if delta < tolerant:
            break
    return v


# 策略改进
def improve_policy(env, v, policy, gamma=1.):
    optimal = True
    for s in range(env.unwrapped.nS):
        q = v2q(env, v, s, gamma)
        a = np.argmax(q)
        if policy[s][a] != 1:
            optimal = False
            policy[s] = 0.
            policy[s][a] = 1.
    return optimal


# 策略迭代的实现
def iterate_policy(env, gamma=1., tolerant=1e-6):
    policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant)
        if improve_policy(env, v, policy):
            break
    return policy, v


# 价值迭代的实现
def iterate_value(env, gamma=1, tolerant=1e-6):
    v = np.zeros(env.unwrapped.nS)
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vmax = max(v2q(env, v, s, gamma))
            delta = max(delta, abs(v[s]-vmax))
            v[s] = vmax
            if delta < tolerant:
                break

    policy = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
    for s in range(env.unwrapped.nS):
        a = np.argmax(v2q(env, v, s, gamma))
        policy[s][a] = 1.
    return policy, v


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped

    random_policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA

    episode_rewards = [play_policy(env, random_policy) for _ in range(100)]
    print("随机策略 平均奖励 ：{}".format(np.mean(episode_rewards)))

    print('状态价值函数：')
    v_random = evaluate_policy(env, random_policy)
    print(v_random.reshape(4, 4))

    print('动作价值函数：')
    q_random = v2q(env, v_random)
    print(q_random)

    policy = random_policy.copy()
    optimal = improve_policy(env, v_random, policy)
    if optimal:
        print('无更新， 最优策略为：')
    else:
        print('有更新，最优策略为：')
    print(policy)

    # 利用策略迭代求解最优策略
    policy_pi, v_pi = iterate_policy(env)
    print('状态价值函数 = ')
    print(v_pi.reshape(4, 4))
    print('最优策略 = ')
    print(np.argmax(policy_pi, axis=1).reshape(4, 4))
    print('价值迭代 平均奖励:{}'.format(play_policy(env, policy_pi)))