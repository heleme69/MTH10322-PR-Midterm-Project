import numpy as np

def multinomial_log_pdf(X, mean, var=None, alpha=0.01):
    """
    Xác suất log cho phân phối Multinomial với làm mịn Laplace alpha.
    theta = (mean + alpha) / sum(mean + alpha)
    """
    theta = (mean + alpha) / (mean + alpha).sum()
    return X @ np.log(theta)

class NaiveBayes:
    """Mô hình Naive Bayes from scratch dùng hàm log_pdf."""
    def __init__(self, log_pdf=multinomial_log_pdf, alpha=0.01):
        self.log_pdf = log_pdf
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.means_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / len(X)
            # Với Multinomial: mean là tổng số đếm pixel của từng vị trí trong lớp c
            self.means_[c] = X_c.sum(axis=0)
        return self

    def predict(self, X):
        log_posteriors = []
        for c in self.classes_:
            prior_term = np.log(self.priors_[c])
            likelihood_term = self.log_pdf(X, self.means_[c], None, alpha=self.alpha)
            log_posteriors.append(prior_term + likelihood_term)
            
        # log_posteriors shape: (n_classes, n_samples) -> chuyển vị lấy argmax
        log_posteriors = np.array(log_posteriors).T
        return self.classes_[np.argmax(log_posteriors, axis=1)]