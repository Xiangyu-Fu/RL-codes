from unityagents import UnityEnvironment
import numpy as np
from P1_Navigation_Agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt
import os
import glob


def main(n_episode=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    env = UnityEnvironment(file_name="E:/2020/RL/Banana_Windows_x86_64/Banana.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    # train
    agent = Agent(state_size=len(state), action_size=action_size, seed=1)
    scores = []
    scores_window = deque(maxlen=10)
    eps = eps_start

    # Loading recent models
    tmp_dirs = glob.glob('checkpoint_P1_eps_*.pth')
    max_round = 1
    for tmp_dir in tmp_dirs:
        file_name = int(tmp_dir[18:-4])
        if file_name > max_round:
            max_round = file_name
    if max_round is not 1:
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint_P1_eps_{}.pth'.format(max_round)))
        agent.qnetwork_target.load_state_dict(torch.load('checkpoint_P1_eps_{}.pth'.format(max_round)))
        scores = np.load("scores_{}.npy".format(max_round)).tolist()

    MODE = 'run'

    if MODE == 'train':
        for i_episode in range(max_round, n_episode + 1):
            score = 0

            # get the default brain
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state
            for t in range(max_t):
                action = agent.act(state, eps)
                env_info = env.step(action)[brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                agent.step(state, action, reward, next_state, done)
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)
            print('\rEpisode {} \t score :{} \t Average Score:{:.2f} \t loss:{}'.format(
                i_episode, score, np.mean(scores_window), agent.loss.cpu().detach().numpy()), end="")
            if i_episode % 50 == 0:
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_P1_eps_{}.pth'.format(i_episode))
                np.save("scores_{}.npy".format(i_episode), scores)
            if np.mean(scores_window) >= 13.0:
                print('\nEnvironment solved in {:d} episode! \t Average Score: {:.2f}'.format(i_episode,
                                                                                              np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    elif MODE == 'run':
        score = 0  # initialize the score
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        while True:
            # action = np.random.randint(action_size)  # select an action
            action = agent.act(state,0.3)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        print("Score: {}".format(score))
        # total steps are 300, that means the agent should collect al least 13 bananas in 300 steps.
        # the first four numbers of array is the speed of the Agent
        env.close()



if __name__ == "__main__":
    main()