import numpy as np

def prepro(observation):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    observation = observation[35:195]  # crop
    observation = observation[::2, ::2, 0]  # downsample by factor of 2
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
    observation_1d = [[0]*len(observation[0])]*len(observation)
    for k in range(len(observation)):
        for i in range(len(observation[k])):
            observation_1d[k][i] = [observation[k][i]]
    return np.array(observation_1d).astype(np.float)

