import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from env import Env
from epsilongreedy import EpsilonGreedy
from ucb import Ucb
from gradientbandit import GradientBandit

# variables
NTEST       = 2000
NSTEP       = 1000
NACTION     = 10

# x-axis range
EPSILON_RANGE = [ 0, 5 ]

UCB_RANGE     = [ 3, 9 ]
GRADIENT_RANGE = [ 2, 9 ]

OPTIMISTIC_RANGE = [ 5, 9 ]
OPTIMISTIC_EPSILON = 0.05

xaxis = np.array([1/128., 1/64., 1/32., 1/16., 1/8., 1/4., 1/2., 1, 2, 4])

# environment to play
env = Env()



Y_EPSILON = []
Y_UCB = []
Y_GRADIENT = []
Y_OPTIMISTIC = []

# run epsilon-greedy
for x in range(EPSILON_RANGE[0], EPSILON_RANGE[1] + 1):        
    SUM = 0

    for itest in range(NTEST):
        env.initTest(A=NACTION)
        e = EpsilonGreedy(env=env, epsilon=xaxis[x], NSTEP=NSTEP)
        SUM += sum(e.solve()) / NSTEP

        print('\rε = {} test {}                             '.format(xaxis[x], itest + 1), end='')
    print()

    Y_EPSILON.append(SUM / NTEST)

# run Optimistic
for x in range(OPTIMISTIC_RANGE[0], OPTIMISTIC_RANGE[1] + 1):        
    SUM = 0

    for itest in range(NTEST):
        env.initTest(A=NACTION)
        e = EpsilonGreedy(env=env, epsilon=OPTIMISTIC_EPSILON, NSTEP=NSTEP, INITQ=xaxis[x])
        SUM += sum(e.solve()) / NSTEP

        print('\rε = {} Q1 = {} test {}                             '.format(OPTIMISTIC_EPSILON, xaxis[x], itest + 1), end='')
    print()

    Y_OPTIMISTIC.append(SUM / NTEST)

# run ucb
for x in range(UCB_RANGE[0], UCB_RANGE[1] + 1):
    SUM = 0

    for itest in range(NTEST):
        env.initTest(A=NACTION)
        ucb = Ucb(env=env, c=xaxis[x], NSTEP=NSTEP)
        SUM += sum(ucb.solve()) / NSTEP

        print('\rUCB, c = {} test {}                             '.format(xaxis[x], itest + 1), end='')
    print()

    Y_UCB.append(SUM / NTEST)

# run gradient
for x in range(GRADIENT_RANGE[0], GRADIENT_RANGE[1] + 1):
    SUM = 0

    for itest in range(NTEST):
        env.initTest(A=NACTION)
        gra = GradientBandit(env=env, NSTEP=NSTEP, alpha=xaxis[x])
        SUM += sum(gra.solve()) / NSTEP

        print('\rGradient, α = {} test {}                             '.format(xaxis[x], itest + 1), end='')
    print()

    Y_GRADIENT.append(SUM / NTEST)


# Draw graph
plt.ylabel('Average reward after first {} steps'.format(NSTEP))

# Make matplotlib can show fractions in x-axis
# [1/128, 1/64, 1/32, ...] -> [0, 1, 2, ...]
X_tick = np.array([])
for item in xaxis:
    X_tick = np.append(X_tick,Fraction(item).limit_denominator())
plt.xticks(np.arange(len(X_tick)), X_tick)

# plot epsilon-greedy
plt.plot( [ i for i in range(EPSILON_RANGE[0], EPSILON_RANGE[1] + 1) ], Y_EPSILON, label='ε-greedy' )

# plot Optimistic
plt.plot( [ i for i in range(OPTIMISTIC_RANGE[0], OPTIMISTIC_RANGE[1] + 1) ], Y_OPTIMISTIC, label='Optimistic' )

# plot ucb
plt.plot( [ i for i in range(UCB_RANGE[0], UCB_RANGE[1] + 1) ], Y_UCB, label='UCB' )

# plot gradient
plt.plot( [ i for i in range(GRADIENT_RANGE[0], GRADIENT_RANGE[1] + 1) ], Y_GRADIENT, label='Gradient Bandit' )

plt.legend()
plt.show()


