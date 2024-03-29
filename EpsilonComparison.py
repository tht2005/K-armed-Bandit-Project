import sys
import matplotlib.pyplot as plt
import numpy as np

from env import Env
from epsilongreedy import EpsilonGreedy

# variables
NTEST       = 2000
NSTEP       = 1000
NACTION     = 10

# environment to play
env = Env()

# epsilon values to test
epsilon_list                = [ 0, 0.01, 0.05, 0.1, 0.2, 0.5 ]
epsilon_style               = [ 'solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (1, 1)) ]

# np.array history[len(epsilon_list)][NSTEP + 1] = { 0 }
history = [ np.array([0] * (NSTEP + 1), dtype=float) for first_d_size in range(len(epsilon_list)) ]



# the epsilonGreedy runs here
for itest in range(NTEST):
    env.initTest(A=NACTION)

    for i in range(len(epsilon_list)):
        e = EpsilonGreedy(env=env, epsilon=epsilon_list[i], NSTEP=NSTEP)
        reward_history = e.solve()

        history[i] += np.array(reward_history)

    print('\rTEST: ', itest + 1, end='')

print()


# Draw graph
plt.xlabel('Time steps')
plt.ylabel('Total Reward')

step = [ i for i in range(NSTEP + 1) ]
for i in range(len(epsilon_list)):
    plt.plot(step, history[i] / NTEST, label="e = {}".format(epsilon_list[i]), linestyle=epsilon_style[i])

plt.legend()
plt.show()

