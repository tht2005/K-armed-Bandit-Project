import numpy as np

class Ucb:
    def __init__(self, env, c, NSTEP):
        self.c              = c
        self.NSTEP          = NSTEP
        self.env            = env

        self.t = 1

        self.A              = env.A
        self.Q              = [0] * env.A
        self.N              = [0] * env.A

    def calc(self, t, a):
        if self.N[a] == 0:
            return np.inf
        return self.Q[a] + self.c * np.sqrt(np.log(t) / self.N[a])

    def chooseAction(self, t):
        return np.argmax([self.calc(t, a) for a in range(self.A)])

    def updateAction(self, a, R):
        self.N[a] = self.N[a] + 1
        self.Q[a] = self.Q[a] + (R - self.Q[a]) / self.N[a]

    def solve(self):
        total_reward = 0
        history = [ 0 ]

        for t in range(1, self.NSTEP + 1):
            a = self.chooseAction(t)
            R = self.env.sendAction(a)
            self.updateAction(a, R)

            total_reward += R
            history.append(total_reward)

        return history

