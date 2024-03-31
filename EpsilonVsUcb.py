import numpy as np
import matplotlib.pyplot as plt

from epsilongreedy import EpsilonGreedy
from ucb import Ucb
from env import Env

#constants
NTEST = 2000
NSTEP = 1000
NACTION = 10

UCB_C = 2

# environment
env = Env()

# history
history = [ np.array([0] * NSTEP, dtype=float) for first_d_size in range(2) ]

for itest in range(NTEST):
    env.initTest(NACTION)

    e = EpsilonGreedy(env=env, epsilon=0.1, NSTEP=NSTEP)
    history[0] += np.array(e.solve())

    u = Ucb(env=env, c=UCB_C, NSTEP=NSTEP)
    history[1] += np.array(u.solve())

    print('\rTEST: ', itest + 1, end='')

print()


# Draw graph
plt.xlabel('Time steps')
plt.ylabel('Average Reward')

step = [ i for i in range(NSTEP) ]
plt.plot(step, history[0] / NTEST, label='ε = {}'.format(0.1))
plt.plot(step, history[1] / NTEST, label='UCB c = {}'.format(UCB_C))

plt.legend()
plt.show()

