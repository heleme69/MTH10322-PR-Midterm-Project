import numpy as np


class SoftmaxRegression:
    """Softmax Regression với mini-batch SGD và L2 regularization."""

    def __init__(self, lr=0.1, max_epochs=20, batch_size=128, lam=1e-3, seed=42):
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lam = lam
        self.seed = seed

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _cross_entropy(self, logits, y):
        """Cross-Entropy từ logits, dùng log-sum-exp để tránh log(0)."""
        z = logits - logits.max(axis=1, keepdims=True)
        return np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y])

    def _loss_and_grad(self, X, y):
        """Loss và gradient trên một batch; L2 chỉ áp dụng cho W."""
        N = len(X)
        logits = X @ self.W + self.b
        probs = self._softmax(logits)
        diff = probs
        diff[np.arange(N), y] -= 1.0
        diff /= N
        grad_W = X.T @ diff + self.lam * self.W
        grad_b = diff.sum(axis=0)
        loss = self._cross_entropy(logits, y) + 0.5 * self.lam * np.sum(self.W ** 2)
        return loss, grad_W, grad_b

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        N, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self.W = np.zeros((d, K), dtype=np.float64)
        self.b = np.zeros(K, dtype=np.float64)

        self.loss_history = []
        for _ in range(self.max_epochs):
            indices = rng.permutation(N)
            for start in range(0, N, self.batch_size):
                idx = indices[start:start + self.batch_size]
                _, grad_W, grad_b = self._loss_and_grad(X[idx], y[idx])
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            loss, _, _ = self._loss_and_grad(X, y)
            self.loss_history.append(float(loss))
        return self

    def predict(self, X):
        logits = X @ self.W + self.b
        return self.classes_[np.argmax(logits, axis=1)]
