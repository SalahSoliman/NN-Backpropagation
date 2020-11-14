import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N+1,)/ np.sqrt(N)
        # print(self.W.shape)
        self.alpha = alpha
    def step(self, x):
        return 1 if x > 0 else 0
    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]
        # print(X.shape)
        for epoch in np.arange(0, epochs):
            for (x, target) in zip (X, y):
                p = self.step(np.dot(x, self.W))
                if p!=target:
                    error = p - target

                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        # ensure our input is a matrix
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))



X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

print("[INFO] Training perceptron")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

print("[INFO] testing Perceptron...")

for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, groud-truth={}, pred={}".format(x, target[0], pred))