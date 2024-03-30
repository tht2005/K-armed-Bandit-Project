import numpy as np
import sys

class Env:
    def initTest(self, A):
        self.A          = A

        self.LOW        = -5
        self.HIGH       = 1
        self.MAXRANGE   = 7

        self.EV         = [ np.random.uniform(low=self.LOW, high=self.HIGH) for i in range(A) ]
        self.RANGE      = [ np.random.uniform(low=0, high=self.MAXRANGE) for i in range(A) ]

    def sendAction(self, a):
        if a < 0 or a >= self.A:
            print('Undefined action: {}'.format(a))
            sys.exit()

        return np.random.normal(loc=self.EV[a], scale=self.RANGE[a])

