import numpy as np
import matplotlib.pyplot as plt

from env import Env
from gradientbandit import GradientBandit

# Const
NTEST = 2000
NSTEP = 1000
NACTION = 10
ENV_CEN = 4

# Algorithms parameters
NALGO = 4
ALPHA = [ 0.1, 0.4, 0.1, 0.4 ]
BASE  = [ True, True, False, False ]


# Environment
env = Env()

# Graph data
history = [ np.array([0] * NSTEP, dtype=float) for first_d_size in range(NALGO) ]

for itest in range(NTEST):
    env.initTest(A=NACTION, CEN=ENV_CEN)

    for i in range(NALGO):
        G = GradientBandit(env=env, NSTEP=NSTEP, alpha=ALPHA[i], baseline=BASE[i])
        history[i] += np.array(G.solve())

    print('\rTEST: ', itest + 1, end='')

print()

# Draw graph
plt.title('Gradient Bandit Algorithm when q*(a) are choosen near {}'.format(ENV_CEN))
plt.xlabel('Time steps')
plt.ylabel('Average Reward')

for i in range(NALGO):
    plt.plot(np.arange(NSTEP), (history[i] / NTEST).tolist(), label="α = {}, {} baseline".format(ALPHA[i], 'with' if BASE[i] else 'no'))

plt.legend()
plt.show()


