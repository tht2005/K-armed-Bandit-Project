import numpy as np

class EpsilonGreedy:
    def __init__(self, env, epsilon, NSTEP, INITQ=0):
        self.epsilon        = epsilon
        self.NSTEP          = NSTEP

        self.env            = env
        self.A              = env.A
        self.Q              = [INITQ for i in range(self.A)]
        self.N              = [0 for i in range(self.A)]

    def chooseAction(self):
        if np.random.uniform(low=0, high=1) < self.epsilon:
            return np.random.randint(low=0, high=self.A)
        return np.argmax(self.Q)

    def updateAction(self, a, R):
        self.N[a] = self.N[a] + 1
        self.Q[a] = self.Q[a] + (R - self.Q[a]) / self.N[a]

    def solve(self):
        history = []

        for step in range(self.NSTEP):
            a = self.chooseAction()
            R = self.env.sendAction(a)
            self.updateAction(a, R)

            history.append(R)

        return history

