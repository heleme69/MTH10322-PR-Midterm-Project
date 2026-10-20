import numpy as np

def compute_confusion_matrix(y_true, y_pred, n_classes=10):
    """
    Tính ma trận nhầm lẫn C kích thước (n_classes, n_classes).
    Hàng i: nhãn thực tế, Cột j: nhãn dự đoán.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def compute_accuracy(y_true, y_pred):
    """Accuracy = Số mẫu dự đoán đúng / Tổng số mẫu"""
    return np.mean(y_true == y_pred)

def compute_classification_metrics(y_true, y_pred, n_classes=10):
    """
    Tính Precision, Recall, F1 theo chuẩn Macro Average (trọng số các lớp bằng nhau).
    Khớp với tài liệu comp3242 (Section 4.2) và bài lab về các độ đo.
    """
    cm = compute_confusion_matrix(y_true, y_pred, n_classes)
    acc = compute_accuracy(y_true, y_pred)
    
    precisions = []
    recalls = []
    f1s = []
    
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        
        # Tránh chia cho 0
        prec_k = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_k = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_k = 2 * (prec_k * rec_k) / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0
        
        precisions.append(prec_k)
        recalls.append(rec_k)
        f1s.append(f1_k)
        
    return {
        "accuracy": acc,
        "precision_macro": np.mean(precisions),
        "recall_macro": np.mean(recalls),
        "f1_macro": np.mean(f1s),
        "confusion_matrix": cm
    }