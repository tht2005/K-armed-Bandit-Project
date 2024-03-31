import numpy as np

class GradientBandit:
    def __init__(self, env, NSTEP, alpha, baseline=True):
        self.A              = env.A
        self.env            = env
        self.NSTEP          = NSTEP
        self.alpha          = alpha
        self.baseline       = baseline

        self.Rbar           = 0 # Average reward (baseline)
        self.N              = 0 # number of steps played
        self.H              = [ 0 for i in range(self.A) ]

    def calculatePi(self):
        E                   = [ np.power(np.e, self.H[a]) for a in range(self.A) ]
        SumE                = sum(E)
        self.pi             = [ (E[a] / SumE) for a in range(self.A) ]

    def chooseAction(self):
        self.calculatePi()

        random_value = np.random.uniform(low=0,high=1)
        for a in range(self.A):
            random_value -= self.pi[a]
            if random_value <= 0:
                return a
        return self.A - 1
    
    def update(self, a, R):
        # update H
        self.H[a] = self.H[a] + self.alpha * (R - self.Rbar) * (1 - self.pi[a])
        for i in range(self.A):
            if i == a:
                continue
            
            self.H[i] = self.H[i] - self.alpha * (R - self.Rbar) * self.pi[i]

        # update Rbar if "with baseline" type
        if self.baseline:
            self.N = self.N + 1
            self.Rbar = self.Rbar + (R - self.Rbar) / self.N


    def solve(self):
        history = []
        
        for step in range(self.NSTEP):
            a = self.chooseAction()
            R = self.env.sendAction(a)
            self.update(a, R)

            history.append(R)

        return history

