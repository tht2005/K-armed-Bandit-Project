import numpy as np

class Env:
    def initTest(self, N):
        self.A          = N
        self.PROB       = [ ( (p / 2) if (p > 0.5) else p ) for p in np.random.rand(N) ]
        self.R          = np.random.randint(0, 100, size=N)

    def sendAction(self, a):
        if a < 0 or a >= self.A:
            print('Undefined action: {}'.format(a))
            sys.exit()

        random_number = np.random.rand()
        if random_number <= self.PROB[a]:
            return self.R[a] - 15
        return -15
