import numpy as np
from numpy.random import default_rng
from sklearn.datasets import make_classification


class MLP:
    def __init__(self, rng):
        self.W0 = rng.standard_normal((5, 10), dtype=np.float32) / (
            5**0.5
        )  # Kaiming init normal for improved initialization
        self.b0 = rng.standard_normal(10, dtype=np.float32) * 0.01
        self.W1 = rng.standard_normal(10, dtype=np.float32) / (
            10**0.5
        )  # Kaiming init normal
        self.b1 = rng.standard_normal(1, dtype=np.float32) * 0.01

    def sigmoid(self, x: np.ndarray):
        return 1.0 / (1.0 + np.exp(-x))

    def _d_sigmoid(self, x: np.ndarray):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def binary_cross_entropy(self, probs: np.ndarray, y_true: np.ndarray):
        out = np.where(y_true == 1, probs, 1 - probs)

        return -np.log(out).mean(axis=0)

    def _d_binary_cross_entropy(self, probs: np.ndarray, y_true: np.ndarray):
        return (probs - y_true) / (probs * (1 - probs))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5):
        for i in range(epochs):
            h0_preact = X @ self.W0 + self.b0  # (N, 10)
            h0 = self.sigmoid(h0_preact)  # (N, 10)

            h1_preact = h0 @ self.W1 + self.b1  # (N,)
            h1 = self.sigmoid(h1_preact)  # (N,)

            loss = self.binary_cross_entropy(probs=h1, y_true=y)  # (1,)
            print(f"Loss: {loss.item():.5f}")
            dloss = self._d_binary_cross_entropy(probs=h1, y_true=y)  # (N,)
            dh1 = dloss * self._d_sigmoid(
                h1_preact
            )  # Same shape as the layer's output--element-wise mult. here: (N,) * (N,)
            dW1 = h0.T @ dh1  # Derived X.T @ (dL / dZ) where Z is the affine transform
            db1 = np.mean(dh1, axis=0)
            dh0 = (
                dh1.reshape(-1, 1) @ self.W1.reshape(-1, 1).T
            )  # h0 is X1, which is a function f(Z0, W1): (N, 10)

            dW0 = X.T @ dh0  # (5, N) @ (N, 10) => (5, 10)
            db0 = np.mean(dh0, axis=0)

            lr = 0.0001
            self.W0 += -lr * dW0
            self.b0 += -lr * db0
            self.W1 += -lr * dW1
            self.b1 += -lr * db1


if __name__ == "__main__":
    rng = default_rng(1234)

    X, y = make_classification(n_samples=10000, n_features=5, n_classes=2)

    model = MLP(rng=rng)

    model.fit(X, y, epochs=15)
