from unityagents import UnityEnvironment
import numpy as np


def main():
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
    score = 0  # initialize the score
    tag = 0
    while True:
        action = np.random.randint(action_size)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if tag % 100 == 0:
            print(state)
        if done:  # exit loop if episode finished
            break
        tag += 1

    print("Score: {}".format(score))
    # total steps are 300, that means the agent should collect al least 13 bananas in 300 steps.
    # the first four numbers of array is the speed of the Agent
    env.close()


if __name__ == "__main__":
    main()