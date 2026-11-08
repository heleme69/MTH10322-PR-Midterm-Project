import time
import warnings

import numpy as np
import torch
import torch.nn as nn


def softmax_cross_entropy(logits, y):
    """Cross-Entropy ổn định và gradient theo logits: (P - Y) / N."""
    N = logits.shape[0]
    z = logits - logits.max(dim=1, keepdim=True).values
    log_probs = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))
    loss = -log_probs[torch.arange(N), y].mean()

    delta = torch.exp(log_probs)
    delta[torch.arange(N), y] -= 1.0
    delta /= N
    return loss, delta


def linear_backward(layer, a_in, delta):
    dW = delta.T @ a_in
    db = delta.sum(dim=0)
    grad_in = delta @ layer.weight
    return dW, db, grad_in


def relu_backward(a_in, grad):
    return grad * (a_in > 0).float()


class InvertedDropout(nn.Module):
    """Dropout thủ công: giữ kỳ vọng activation khi huấn luyện."""

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0:
            self.mask = None
            return x
        self.mask = (torch.rand_like(x) >= self.p).to(x.dtype) / (1 - self.p)
        return x * self.mask


def forward_with_cache(model, X):
    """Feedforward tự viết; cache dùng cho Backpropagation thủ công."""
    cache = []
    a = X
    for layer in model:
        a_in = a
        if isinstance(layer, nn.Linear):
            a = a_in @ layer.weight.T + layer.bias
        elif isinstance(layer, nn.ReLU):
            a = torch.relu(a_in)
        elif isinstance(layer, InvertedDropout):
            a = layer(a_in)  # Dropout cũng là lớp tự cài ở trên.
        cache.append((layer, a_in, a))
    return a, cache


def backward(cache, grad_out, weight_decay=0.0):
    """Lan truyền ngược thủ công; không dùng autograd khi train."""
    grad = grad_out
    for layer, a_in, a_out in reversed(cache):
        if isinstance(layer, nn.Linear):
            dW, db, grad = linear_backward(layer, a_in, grad)
            layer.weight.grad = dW + weight_decay * layer.weight
            layer.bias.grad = db
        elif isinstance(layer, nn.ReLU):
            grad = relu_backward(a_in, grad)
        elif isinstance(layer, InvertedDropout) and layer.mask is not None:
            grad = grad * layer.mask
    return grad


class SGDMomentum:
    """Mini-batch SGD + Momentum, cập nhật trọng số thủ công."""

    def __init__(self, params, lr, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for p, v in zip(self.params, self.velocity):
                v.mul_(self.momentum).sub_(self.lr * p.grad)
                p.add_(v)


class MLPFromScratch:
    """MLP tự viết Feedforward/Backpropagation, có Dropout và Early Stopping."""

    def __init__(self, hidden_sizes=[128, 64], lr=0.05, momentum=0.9,
                 max_epochs=15, batch_size=128, dropout=0.0,
                 weight_decay=0.0, patience=5, seed=42, eval_batch_size=1024):
        self.sizes = [784] + list(hidden_sizes) + [10]
        self.classes_ = np.arange(10)  # MNIST: neuron k tương ứng nhãn k
        self.lr = lr
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.eval_batch_size = eval_batch_size

        if seed is not None:
            torch.manual_seed(seed)
        layers = []
        for i in range(len(self.sizes) - 1):
            lin = nn.Linear(self.sizes[i], self.sizes[i + 1])
            nn.init.kaiming_normal_(
                lin.weight,
                nonlinearity="linear" if i == len(self.sizes) - 2 else "relu",
            )
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(self.sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(InvertedDropout(dropout))
        self.model = nn.Sequential(*layers)

    def _evaluate(self, X, y):
        """Trả về (Cross-Entropy trung bình, accuracy), xử lý theo batch."""
        total_loss, correct = 0.0, 0
        for start in range(0, len(X), self.eval_batch_size):
            end = start + self.eval_batch_size
            logits, _ = forward_with_cache(self.model, X[start:end])
            loss, _ = softmax_cross_entropy(logits, y[start:end])
            total_loss += float(loss) * len(X[start:end])
            correct += int((logits.argmax(dim=1) == y[start:end]).sum())
        return total_loss / len(X), correct / len(X)

    def _mean_loss(self, X, y):
        """Giữ API cũ để tương thích với code đang dùng."""
        return self._evaluate(X, y)[0]

    def fit(self, X, y, X_val=None, y_val=None):
        if (X_val is None) != (y_val is None):
            raise ValueError("Cần truyền đồng thời X_val và y_val.")

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        if isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if isinstance(y_val, np.ndarray):
            y_val = torch.from_numpy(y_val).long()

        if len(X) != len(y):
            raise ValueError("X và y phải có cùng số mẫu.")
        if X_val is not None and len(X_val) != len(y_val):
            raise ValueError("X_val và y_val phải có cùng số mẫu.")

        if self.seed is not None:
            torch.manual_seed(self.seed)
        optimizer = SGDMomentum(self.model.parameters(), self.lr, self.momentum)

        # Giữ nguyên train_loss v1.3 = Cross-Entropy + L2.
        # epoch_time chỉ đo bước huấn luyện, không tính thời gian đánh giá.
        self.history_ = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "epoch_time": [],
        }
        self.best_epoch_ = None
        self.stopped_early_ = False
        self.diverged_ = False
        self.fit_time_ = 0.0
        self.eval_time_ = 0.0
        best_loss = float("inf")
        best_weights = None
        waiting = 0

        with torch.no_grad():
            for epoch in range(self.max_epochs):
                t0 = time.perf_counter()
                self.model.train()
                perm = torch.randperm(len(X))
                for start in range(0, len(X), self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    logits, cache = forward_with_cache(self.model, X[idx])
                    loss, delta = softmax_cross_entropy(logits, y[idx])
                    if not torch.isfinite(loss):
                        self.diverged_ = True
                        break
                    backward(cache, delta, weight_decay=self.weight_decay)
                    optimizer.step()

                epoch_seconds = time.perf_counter() - t0
                self.fit_time_ += epoch_seconds
                if self.diverged_:
                    warnings.warn(
                        "Loss NaN/Inf: dừng huấn luyện; thử giảm learning rate.",
                        RuntimeWarning,
                    )
                    break

                # Đánh giá ở chế độ eval: Dropout được tắt.
                t1 = time.perf_counter()
                self.model.eval()
                train_ce, train_acc = self._evaluate(X, y)
                reg = 0.5 * self.weight_decay * sum(
                    float((layer.weight ** 2).sum())
                    for layer in self.model if isinstance(layer, nn.Linear)
                )
                train_loss = train_ce + reg
                if not np.isfinite(train_loss):
                    self.diverged_ = True
                    self.eval_time_ += time.perf_counter() - t1
                    warnings.warn(
                        "Train loss NaN/Inf: dừng huấn luyện.",
                        RuntimeWarning,
                    )
                    break

                self.history_["train_loss"].append(train_loss)
                self.history_["train_acc"].append(train_acc)
                self.history_["epoch_time"].append(epoch_seconds)

                if X_val is not None:
                    val_loss, val_acc = self._evaluate(X_val, y_val)
                    self.history_["val_loss"].append(val_loss)
                    self.history_["val_acc"].append(val_acc)
                    if not np.isfinite(val_loss):
                        self.diverged_ = True
                        self.eval_time_ += time.perf_counter() - t1
                        warnings.warn(
                            "Validation loss NaN/Inf: dừng huấn luyện.",
                            RuntimeWarning,
                        )
                        break
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self.best_epoch_ = epoch + 1
                        best_weights = [p.clone() for p in self.model.parameters()]
                        waiting = 0
                    else:
                        waiting += 1
                        if self.patience is not None and waiting >= self.patience:
                            self.stopped_early_ = True
                            self.eval_time_ += time.perf_counter() - t1
                            break

                self.eval_time_ += time.perf_counter() - t1

            if best_weights is not None and not self.diverged_:
                for p, best in zip(self.model.parameters(), best_weights):
                    p.copy_(best)

        self.epochs_trained_ = len(self.history_["train_loss"])
        self.model.eval()
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(X), self.eval_batch_size):
                logits, _ = forward_with_cache(
                    self.model, X[start:start + self.eval_batch_size]
                )
                predictions.append(logits.argmax(dim=1).cpu().numpy())
        indices = np.concatenate(predictions)
        return self.classes_[indices]
