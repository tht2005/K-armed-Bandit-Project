import numpy as np
import matplotlib.pyplot as plt

from env import Env
from epsilongreedy import EpsilonGreedy

#constants
NTEST = 2000
NSTEP = 1000
NACTION = 10

EPSILON1 = 0.1

INITQ = 5
EPSILON2 = 0

# environment
env = Env()

# history
history = [ np.array([0] * NSTEP, dtype=float) for first_d_size in range(2) ]

for itest in range(NTEST):
    env.initTest(NACTION)

    e = EpsilonGreedy(env=env, epsilon=EPSILON1, NSTEP=NSTEP)
    history[0] += np.array(e.solve())

    o = EpsilonGreedy(env=env, epsilon=EPSILON2, NSTEP=NSTEP, INITQ=INITQ)
    history[1] += np.array(o.solve())

    print('\rTEST: ', itest + 1, end='')
print()

history[0] /= NTEST
history[1] /= NTEST

# Draw graph
plt.xlabel('Time steps')
plt.ylabel('Average Reward')

step = [ i for i in range(NSTEP) ]
plt.plot(step, history[0].tolist(), label='ε = {}'.format(EPSILON1))
plt.plot(step, history[1].tolist(), label='Q1 = {}, ε = {}'.format(INITQ, EPSILON2))

plt.legend()
plt.show()
