# Open AI: https://gym.openai.com/envs/
# Gym Github: https://github.com/openai/gym
# Youtube video: https://www.youtube.com/watch?v=3zeg7H6cAJw&index=59&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
# Youtube video: https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/

import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter


LR = 1e0-3 # Learning Rate

env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    for episode in range (5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range (goal_steps):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break

# some_random_games_first()

def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in range(goal_steps)
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0,2)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                env.reset()
                break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                # saving our training data
                training_data.append([data[0], output])

            # reset env to play again
            env.reset()
            # save overall scores
            scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Accepted score:', accepted_scores)
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

initial_population()




