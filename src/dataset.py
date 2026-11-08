import gzip
import os
import shutil
import urllib.request

import numpy as np
import torchvision

# Bản sao 4 file MNIST gốc trên GitHub, dùng khi nguồn mặc định của torchvision không truy cập được.
# Cách làm này lấy từ hàm load_mnist() của Lab 04 (BTVN Tuần 4).
MNIST_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
               "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
GITHUB_MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"


def _download_from_mirror(root):
    raw = os.path.join(root, "MNIST", "raw")
    os.makedirs(raw, exist_ok=True)
    for name in MNIST_FILES:
        dst = os.path.join(raw, name)
        if os.path.exists(dst):
            continue
        gz = dst + ".gz"
        urllib.request.urlretrieve(GITHUB_MIRROR + name + ".gz", gz)
        with gzip.open(gz, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def _load_raw_mnist(data_dir):
    """Thử tải bằng torchvision; nếu lỗi thì chuyển sang bản sao trên GitHub."""
    try:
        train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
        test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    except Exception as e:
        print("Nguồn mặc định của torchvision lỗi (", type(e).__name__, "), chuyển sang mirror GitHub")
        _download_from_mirror(data_dir)
        train = torchvision.datasets.MNIST(root=data_dir, train=True, download=False)
        test = torchvision.datasets.MNIST(root=data_dir, train=False, download=False)
    return train, test


def load_and_preprocess_mnist(data_dir="../data"):
    """
    Tải và tiền xử lý bộ dữ liệu MNIST chuẩn (70.000 ảnh 28x28).

    Quy trình:
    1. Tải tập Train (60.000 ảnh) và Test (10.000 ảnh).
    2. Chuyển đổi tensor sang NumPy, làm phẳng 28x28 -> 784 chiều.
    3. Chuẩn hóa pixel về đoạn [0.0, 1.0].
    4. Tách tập 60.000 mẫu thành 50.000 mẫu Train và 10.000 mẫu Validation.

    Trả về:
    - X_train (50000, 784), y_train (50000,)
    - X_val   (10000, 784), y_val   (10000,)
    - X_test  (10000, 784), y_test  (10000,)
    """
    os.makedirs(data_dir, exist_ok=True)

    # Tải dữ liệu thô (có phương án dự phòng khi nguồn mặc định lỗi)
    train_raw, test_raw = _load_raw_mnist(data_dir)

    # 1. Trích xuất mảng numpy và làm phẳng (Flatten: 28x28 -> 784)
    # train_raw.data có kiểu uint8 miền [0, 255]
    X_train_full = train_raw.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train_full = train_raw.targets.numpy().astype(np.int64)

    X_test = test_raw.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test = test_raw.targets.numpy().astype(np.int64)

    # 2. Xáo trộn ngẫu nhiên tập 60.000 mẫu và tách Train (50k) / Validation (10k)
    # Cố định seed 42 để kết quả có thể tái lập (Reproducibility)
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X_train_full))

    train_idx = indices[:50000]
    val_idx = indices[50000:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
