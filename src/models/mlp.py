import torch
import torch.nn as nn
import numpy as np

# Tắt autograd để toàn bộ gradient do ta tự tính
torch.set_grad_enabled(False)

def softmax_cross_entropy(logits, y):
    """Tính loss cross-entropy và delta ở lớp ra: BP1 = (P - Y) / N"""
    N = logits.shape[0]
    z = logits - logits.max(dim=1, keepdim=True).values
    log_probs = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))
    loss = -log_probs[torch.arange(N), y].mean()
    
    Y = torch.zeros_like(logits)
    Y[torch.arange(N), y] = 1.0
    delta = (torch.exp(log_probs) - Y) / N
    return loss, delta

def linear_backward(layer, a_in, delta):
    """Lan truyền ngược qua tầng tuyến tính: tính dW, db và gradient đầu vào"""
    dW = delta.T @ a_in
    db = delta.sum(dim=0)
    grad_in = delta @ layer.weight
    return dW, db, grad_in

def relu_backward(a_in, grad):
    """Đạo hàm của hàm kích hoạt ReLU"""
    return grad * (a_in > 0).float()

def forward_with_cache(model, X):
    """Forward pass và lưu lại cache các tầng để tính backward"""
    cache = []
    a = X
    for layer in model:
        a_in = a
        a = layer(a_in)
        cache.append((layer, a_in, a))
    return a, cache

def backward(cache, grad_out, weight_decay=0.0):
    """Duyệt ngược cache và gán thủ công .grad cho từng trọng số"""
    grad = grad_out
    for layer, a_in, a_out in reversed(cache):
        if isinstance(layer, nn.Linear):
            dW, db, grad = linear_backward(layer, a_in, grad)
            layer.weight.grad = dW + weight_decay * layer.weight
            layer.bias.grad = db
        elif isinstance(layer, nn.ReLU):
            grad = relu_backward(a_in, grad)
    return grad

class SGDMomentum:
    """Bộ tối ưu SGD có quán tính (momentum) cập nhật in-place"""
    def __init__(self, params, lr, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            v.mul_(self.momentum).sub_(self.lr * p.grad)
            p.add_(v)

class MLPFromScratch:
    """Mô hình MLP tự huấn luyện bằng thuật toán lan truyền ngược tự viết"""
    def __init__(self, hidden_sizes=[128, 64], lr=0.05, momentum=0.9, max_epochs=15, batch_size=128):
        self.sizes = [784] + hidden_sizes + [10]
        self.lr = lr
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        
        # Xây dựng kiến trúc nn.Sequential
        layers = []
        for i in range(len(self.sizes) - 1):
            lin = nn.Linear(self.sizes[i], self.sizes[i+1])
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(self.sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def fit(self, X, y):
        # Đảm bảo dữ liệu ở dạng torch.Tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).long()
            
        N = len(X)
        optimizer = SGDMomentum(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        
        for epoch in range(self.max_epochs):
            perm = torch.randperm(N)
            for s in range(0, N, self.batch_size):
                idx = perm[s:s + self.batch_size]
                X_batch, y_batch = X[idx], y[idx]
                
                # 1. Forward
                logits, cache = forward_with_cache(self.model, X_batch)
                # 2. Loss & delta
                loss, delta = softmax_cross_entropy(logits, y_batch)
                # 3. Backward tự tính
                backward(cache, delta)
                # 4. Cập nhật
                optimizer.step()
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        logits, _ = forward_with_cache(self.model, X)
        return logits.argmax(dim=1).numpy()