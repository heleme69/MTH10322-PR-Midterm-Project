
"""Unit tests cho bốn model v1.3 và MLP Refine Lab-aligned.

Chạy từ thư mục gốc project_midterm:
    python -m pytest tests/test_models.py -v
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.naive_bayes import NaiveBayes
from src.models.softmax_regression import SoftmaxRegression
from src.models.mlp import (
    MLPFromScratch, SGDMomentum, InvertedDropout,
    softmax_cross_entropy, forward_with_cache, backward,
)
from src.models.ensemble import RandomForestFromScratch


def test_naive_bayes():
    X = np.array([[3., 0.], [2., 1.], [0., 3.], [1., 2.]])
    y = np.array([0, 0, 1, 1])
    model = NaiveBayes(alpha=1.0).fit(X, y)
    assert model.priors_[0] == model.priors_[1] == 0.5
    np.testing.assert_array_equal(model.means_[0], [5., 1.])
    np.testing.assert_array_equal(model.means_[1], [1., 5.])
    np.testing.assert_array_equal(model.predict(np.array([[3., 0.], [0., 3.]])), [0, 1])


def test_softmax_gradient():
    """So sánh gradient thật trong model với sai phân hữu hạn, gồm L2."""
    X = np.array([[0.2, -0.4], [1.1, 0.3], [-0.6, 0.8]])
    y = np.array([0, 2, 1])
    model = SoftmaxRegression(lam=0.2)
    model.W = np.array([[0.3, -0.2, 0.1], [0.1, 0.4, -0.3]])
    model.b = np.array([0.1, -0.2, 0.2])
    _, dW, db = model._loss_and_grad(X, y)
    eps = 1e-5
    for param, grad in [(model.W, dW), (model.b, db)]:
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps
            plus = model._loss_and_grad(X, y)[0]
            param[idx] = old - eps
            minus = model._loss_and_grad(X, y)[0]
            param[idx] = old
            np.testing.assert_allclose(grad[idx], (plus - minus) / (2 * eps), rtol=1e-5, atol=1e-7)


def test_softmax_stability():
    model = SoftmaxRegression()
    logits = np.array([[10000., 9999., -10000.], [-10000., 10000., 9999.]])
    y = np.array([0, 1])
    probs = model._softmax(logits)
    assert np.isfinite(model._cross_entropy(logits, y))
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_mlp_gradient():
    """Backprop thủ công so với autograd chỉ trong unit test."""
    torch.manual_seed(4)
    model = MLPFromScratch(hidden_sizes=[5], dropout=0.0, weight_decay=0.07)
    X = torch.randn(4, 784)
    y = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        logits, cache = forward_with_cache(model.model, X)
        _, delta = softmax_cross_entropy(logits, y)
        backward(cache, delta, weight_decay=model.weight_decay)
        manual = [p.grad.clone() for p in model.model.parameters()]
    with torch.enable_grad():
        logits = model.model(X)
        loss = nn.functional.cross_entropy(logits, y)
        loss += 0.5 * model.weight_decay * sum(
            (layer.weight ** 2).sum() for layer in model.model if isinstance(layer, nn.Linear)
        )
        reference = torch.autograd.grad(loss, list(model.model.parameters()))
    # Feedforward tự viết phải cho cùng logits với nn.Sequential tham chiếu.
    with torch.no_grad():
        manual_logits, _ = forward_with_cache(model.model, X)
    torch.testing.assert_close(manual_logits, logits.detach())
    for actual, expected in zip(manual, reference):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_sgd_momentum():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = SGDMomentum([param], lr=0.1, momentum=0.9)
    param.grad = torch.tensor([2.0])
    optimizer.step()  # v1=-0.2; w1=0.8
    torch.testing.assert_close(param, torch.tensor([0.8]))
    param.grad = torch.tensor([2.0])
    optimizer.step()  # v2=-0.38; w2=0.42 (đúng bài Lab tuần 4)
    torch.testing.assert_close(param, torch.tensor([0.42]))


def test_dropout():
    torch.manual_seed(7)
    layer = InvertedDropout(p=0.5)
    X = torch.ones(8, 8)
    layer.train()
    output = layer(X)
    assert set(output.unique().tolist()).issubset({0.0, 2.0})
    torch.testing.assert_close(output, X * layer.mask)
    # Backward phải dùng đúng mask của Forward.
    cache = [(layer, X, output)]
    torch.testing.assert_close(backward(cache, torch.ones_like(X)), layer.mask)
    layer.eval()
    torch.testing.assert_close(layer(X), X)


def test_early_stopping():
    torch.manual_seed(5)
    # Train khuyến khích lớp 0; validation cùng X nhưng nhãn 1 -> loss tăng.
    X = np.zeros((8, 784), dtype=np.float32)
    X[:, 0] = 1.0
    y_train = np.zeros(8, dtype=np.int64)
    y_val = np.ones(8, dtype=np.int64)
    model = MLPFromScratch(hidden_sizes=[4], lr=0.05, momentum=0.0,
                           max_epochs=12, batch_size=8, patience=2,
                           eval_batch_size=3, seed=5)
    model.fit(X, y_train, X, y_val)
    assert model.stopped_early_
    assert model.best_epoch_ == 1
    assert model.epochs_trained_ == 3
    assert model.history_["val_loss"][-1] > model.history_["val_loss"][0]
    with torch.no_grad():
        logits, _ = forward_with_cache(model.model, torch.from_numpy(X))
        loss, _ = softmax_cross_entropy(logits, torch.from_numpy(y_val))
    np.testing.assert_allclose(float(loss), model.history_["val_loss"][0], rtol=1e-6)


def test_forest_voting():
    class StubTree:
        def __init__(self, values):
            self.values = np.array(values)

        def predict(self, X):
            return self.values

    forest = RandomForestFromScratch(n_estimators=3)
    forest.trees = [StubTree([1, 2, 1]), StubTree([1, 3, 2]), StubTree([2, 3, 3])]
    # Mẫu thứ ba hòa phiếu 1-2-3: np.argmax chọn nhãn nhỏ nhất (1).
    np.testing.assert_array_equal(forest.predict(np.zeros((3, 2))), [1, 3, 1])


# =================== MLP Refine v1.3: 4 kiểm thử bổ sung ===================

def test_mlp_classes():
    """classes_ phản ánh nhãn MNIST; predict thực sự ánh xạ qua classes_."""
    model = MLPFromScratch(hidden_sizes=[4], seed=42)
    np.testing.assert_array_equal(model.classes_, np.arange(10))

    # Ép neuron thứ 7 có logit lớn nhất để kiểm tra predict có dùng classes_.
    with torch.no_grad():
        final_linear = next(layer for layer in reversed(model.model)
                            if isinstance(layer, nn.Linear))
        final_linear.weight.zero_()
        final_linear.bias.fill_(-10.0)
        final_linear.bias[7] = 10.0

    X = np.zeros((3, 784), dtype=np.float32)
    np.testing.assert_array_equal(model.predict(X), [7, 7, 7])

    # Đổi bảng ánh xạ chỉ trong test để chắc chắn predict không bỏ qua classes_.
    model.classes_ = np.arange(10)[::-1]
    np.testing.assert_array_equal(model.predict(X), [2, 2, 2])


def test_mlp_history():
    """History đủ 5 chỉ số; train_loss vẫn bằng CE + L2 như v1.3."""
    X = np.zeros((10, 784), dtype=np.float32)
    X[:, 0] = np.linspace(0.1, 1.0, len(X))
    y = np.arange(10, dtype=np.int64)
    model = MLPFromScratch(
        hidden_sizes=[4], lr=0.01, momentum=0.0,
        max_epochs=1, batch_size=5, weight_decay=0.01,
        patience=2, seed=11, eval_batch_size=3,
    )
    assert model.fit(X, y, X, y) is model
    assert model.epochs_trained_ == 1
    assert set(model.history_) == {
        "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"
    }
    assert all(len(series) == 1 for series in model.history_.values())
    assert all(np.isfinite(series).all() for series in model.history_.values())
    assert 0.0 <= model.history_["train_acc"][0] <= 1.0
    assert 0.0 <= model.history_["val_acc"][0] <= 1.0
    assert model.history_["epoch_time"][0] >= 0.0
    assert model.fit_time_ >= 0.0 and model.eval_time_ >= 0.0
    np.testing.assert_allclose(model.fit_time_, sum(model.history_["epoch_time"]))

    # Chỉ có một epoch nên best weights chính là weights hiện tại.
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    with torch.no_grad():
        ce, acc = model._evaluate(X_t, y_t)
        l2 = 0.5 * model.weight_decay * sum(
            float((layer.weight ** 2).sum())
            for layer in model.model if isinstance(layer, nn.Linear)
        )
    np.testing.assert_allclose(model.history_["train_loss"][0], ce + l2, rtol=1e-6)
    np.testing.assert_allclose(model.history_["val_loss"][0], ce, rtol=1e-6)
    np.testing.assert_allclose(model.history_["train_acc"][0], acc, rtol=1e-6)
    np.testing.assert_allclose(model.history_["val_acc"][0], acc, rtol=1e-6)


def test_mlp_divergence():
    """NaN trong logits phải khiến fit dừng và đánh dấu diverged_."""
    X = np.zeros((4, 784), dtype=np.float32)
    X[0, 0] = np.nan
    y = np.zeros(4, dtype=np.int64)
    model = MLPFromScratch(
        hidden_sizes=[4], max_epochs=3, batch_size=4, seed=42,
    )
    with pytest.warns(RuntimeWarning, match="NaN/Inf"):
        model.fit(X, y)
    assert model.diverged_
    assert model.epochs_trained_ == 0
    assert model.history_["train_loss"] == []


def test_mlp_validation_pair():
    """Validation cần có đủ X_val và y_val, tránh bị bỏ qua âm thầm."""
    X = np.zeros((4, 784), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)
    model = MLPFromScratch(hidden_sizes=[4], max_epochs=1)
    with pytest.raises(ValueError, match="đồng thời"):
        model.fit(X, y, X_val=X)
    with pytest.raises(ValueError, match="đồng thời"):
        model.fit(X, y, y_val=y)
