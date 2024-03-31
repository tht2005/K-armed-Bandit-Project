import numpy as np
import sys

class Env:
    def initTest(self, A, CEN=0):
        self.A          = A
        self.EV         = [ np.random.normal(loc=CEN, scale=1) for i in range(A) ]

    def sendAction(self, a):
        if a < 0 or a >= self.A:
            print('Undefined action: {}'.format(a))
            sys.exit()

        return np.random.normal(loc=self.EV[a], scale=1)

