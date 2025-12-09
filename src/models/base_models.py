# src/models/base_models.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class BaseModel:
    """
    统一接口：所有模型都实现 train / predict / predict_proba
    方便后续集成与评估。
    """

    def train(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LogisticRegressionModel(BaseModel):
    """优化后的逻辑回归"""

    def __init__(self, C: float = 0.5):
        self.model = LogisticRegression(
            C=C,
            class_weight='balanced',
            solver='liblinear',
            penalty='l2',
            random_state=42,
            max_iter=1000
        )

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class DecisionTreeModel(BaseModel):
    """决策树"""

    def __init__(self, max_depth: int = 7):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class SVMModel(BaseModel):
    """SVM"""

    def __init__(self, C: float = 2.0, gamma: float = 0.1):
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            cache_size=500,  # 提高训练速度
            random_state=42,
        )

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class NaiveBayesModel(BaseModel):
    """高斯朴素贝叶斯"""

    def __init__(self):
        self.model = GaussianNB()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)