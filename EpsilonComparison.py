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
epsilon_list                = [ 0, 0.05, 0.1, 0.3 ]
epsilon_style               = [ 'solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (1, 1)) ]

# np.array history[len(epsilon_list)][NSTEP] = { 0 }
history = [ np.array([0] * NSTEP, dtype=float) for first_d_size in range(len(epsilon_list)) ]



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
plt.title('ε-greedy with different ε parameters')
plt.xlabel('Time steps')
plt.ylabel('Average Reward')

for i in range(len(epsilon_list)):
    plt.plot(np.arange(NSTEP), (history[i] / NTEST).tolist(), label="ε = {}".format(epsilon_list[i]))

plt.legend()
plt.show()

