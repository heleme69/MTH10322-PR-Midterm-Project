import numpy as np

class SoftmaxRegression:
    """
    Phân lớp tuyến tính Softmax Regression (Multinomial Logistic Regression).
    Huấn luyện bằng Mini-batch Gradient Descent theo lý thuyết comp3242 (Ch.2).
    """
    def __init__(self, lr=0.1, max_epochs=20, batch_size=128, lam=1e-3, seed=42):
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lam = lam  # Điều chuẩn L2 (weight decay)
        self.seed = seed

    def _softmax(self, z):
        """Softmax trừ z_max để ổn định số học chống tràn số (comp3242 mục 7.1)"""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        N, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        
        # Khởi tạo trọng số W (d, K) và bias b (K,)
        self.W = np.zeros((d, K), dtype=np.float32)
        self.b = np.zeros(K, dtype=np.float32)
        
        # Mã hóa nhãn One-hot
        Y_onehot = np.zeros((N, K), dtype=np.float32)
        Y_onehot[np.arange(N), y] = 1.0

        self.loss_history = []
        for epoch in range(self.max_epochs):
            indices = rng.permutation(N)
            for start_idx in range(0, N, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_idx]
                Y_batch = Y_onehot[batch_idx]
                B = len(X_batch)
                
                # 1. Forward pass
                logits = X_batch @ self.W + self.b
                probs = self._softmax(logits)
                
                # 2. Gradient (grad = (P - Y) / B)
                diff = (probs - Y_batch) / B
                grad_W = X_batch.T @ diff + self.lam * self.W
                grad_b = np.sum(diff, axis=0)
                
                # 3. Cập nhật tham số
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b
                
            # Đánh giá loss ở cuối mỗi epoch
            total_logits = X @ self.W + self.b
            total_probs = self._softmax(total_logits)
            loss = -np.mean(np.log(total_probs[np.arange(N), y] + 1e-12))
            self.loss_history.append(loss)
            
        return self

    def predict(self, X):
        logits = X @ self.W + self.b
        return self.classes_[np.argmax(logits, axis=1)]