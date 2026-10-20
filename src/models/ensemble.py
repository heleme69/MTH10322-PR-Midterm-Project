import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestFromScratch:
    """
    Ensemble Random Forest cài đặt nguyên lý Bootstrap Aggregation (Bagging)
    kết hợp Subsampling đặc trưng theo lý thuyết.
    """
    def __init__(self, n_estimators=50, max_depth=15, max_features="sqrt", seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        N = len(X)
        self.trees = []
        
        for i in range(self.n_estimators):
            # 1. Lấy mẫu có hoàn lại (Bootstrap sample)
            boot_idx = rng.choice(N, size=N, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            
            # 2. Huấn luyện 1 cây cơ sở với tập đặc trưng con ngẫu nhiên
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.seed + i
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # 3. Biểu quyết đa số (Majority Voting) giữa tất cả các cây
        predictions = np.array([tree.predict(X) for tree in self.trees]) # (n_trees, n_samples)
        
        final_preds = []
        for j in range(X.shape[0]):
            counts = np.bincount(predictions[:, j])
            final_preds.append(np.argmax(counts))
            
        return np.array(final_preds)