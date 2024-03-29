
class EpsilonGreedy:
    def __init__(self, env, epsilon, NSTEP):
        self.epsilon        = epsilon
        self.NSTEP          = NSTEP

        self.env            = env
        self.A              = env.A
        self.Q              = [0] * self.A
        self.N              = [0] * self.A

    def solve(self):
        for step in range(self.NSTEP):
            print(step + 1)
