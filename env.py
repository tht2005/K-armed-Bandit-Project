import numpy as np
import sys

class Env:
    def initTest(self, A):
        self.A          = A

        self.LOW        = -4
        self.HIGH       = 3

        self.E          = [ np.random.dirichlet(np.ones(self.HIGH - self.LOW + 1)) for _ in range(A) ]

    def sendAction(self, a):
        if a < 0 or a >= self.A:
            print('Undefined action: {}'.format(a))
            sys.exit()

        random_value = np.random.rand()
        # reward
        for i in range(len(self.E[a])):
            random_value -= self.E[a][i]
            if random_value <= 0:
                return self.LOW + i
            
        return self.HIGH

