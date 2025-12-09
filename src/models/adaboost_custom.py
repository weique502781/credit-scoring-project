# src/models/adaboost_custom.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import List, Dict


class ImprovedAdaBoost:
    """
    AdaBoost
    1. 使用更强的弱学习器（max_depth=3）
    2. 添加早停机制
    3. 使用bootstrap采样增加多样性
    """

    def __init__(self, n_estimators: int = 100,
                 learning_rate: float = 0.8,
                 weak_learner_depth: int = 3,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learner_depth = weak_learner_depth
        self.random_state = random_state

        self.models: List[DecisionTreeClassifier] = []
        self.alphas: List[float] = []
        self.pos_label = 1

    def _check_binary_labels(self, y: np.ndarray) -> np.ndarray:
        """确保是二分类，并将标签映射到 {-1, +1}"""
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"AdaBoost 目前只支持二分类，收到类别: {classes}")

        self.pos_label = classes.max()
        y_mapped = np.where(y == self.pos_label, 1, -1)
        return y_mapped

    def train(self, X: np.ndarray, y: np.ndarray):
        """训练AdaBoost"""
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]
        y_mapped = self._check_binary_labels(y)

        # 初始化样本权重
        w = np.ones(n_samples) / n_samples

        rng = np.random.RandomState(self.random_state)

        self.models = []
        self.alphas = []

        # 添加早停机制
        best_error = 1.0
        patience = 10
        patience_counter = 0

        for t in range(self.n_estimators):
            # 1. 使用bootstrap采样增加多样性
            indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=w
            )
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # 2. 训练更强的弱学习器
            weak_learner = DecisionTreeClassifier(
                max_depth=self.weak_learner_depth,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=rng.randint(0, 1_000_000)
            )
            weak_learner.fit(X_bootstrap, y_bootstrap)

            # 3. 在整个数据集上预测
            y_pred = weak_learner.predict(X)
            y_pred_mapped = np.where(y_pred == self.pos_label, 1, -1)

            # 4. 计算加权错误率
            misclassified = (y_pred_mapped != y_mapped).astype(float)
            error = np.sum(w * misclassified) / np.sum(w)

            # 5. 如果误差太大，尝试重新训练
            if error >= 0.45:
                print(f"[AdaBoost] 第 {t + 1} 轮误差 {error:.4f} 较高，尝试重新采样")
                continue

            # 6. 计算弱学习器权重
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / (error + 1e-10))

            # 7. 更新样本权重
            w *= np.exp(-alpha * y_mapped * y_pred_mapped)
            w /= np.sum(w)  # 归一化

            # 8. 保存本轮模型
            self.models.append(weak_learner)
            self.alphas.append(alpha)

            print(f"[AdaBoost] 轮数 {t + 1:3d}, error={error:.4f}, alpha={alpha:.4f}")

            # 9. 早停检查
            if error < best_error:
                best_error = error
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and t > 20:
                print(f"[AdaBoost] 早停触发，连续 {patience} 轮无改进")
                break

        print(f"[AdaBoost] 训练完成，共 {len(self.models)} 个弱学习器")

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算加权投票分数"""
        X = np.asarray(X)
        if not self.models:
            raise RuntimeError("请先调用 train() 方法训练模型")

        F = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            y_pred = model.predict(X)
            y_pred_mapped = np.where(y_pred == self.pos_label, 1, -1)
            F += alpha * y_pred_mapped

        # 归一化
        if len(self.alphas) > 0:
            F = F / np.sum(np.abs(self.alphas))

        return F

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        F = self._decision_function(X)
        y_pred_internal = np.where(F >= 0, 1, -1)
        return np.where(y_pred_internal == 1, self.pos_label, 1 - self.pos_label)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        F = self._decision_function(X)

        F_scaled = np.tanh(F)
        # p_pos = (F_scaled + 1) / 2
        p_pos = 1 / (1 + np.exp(-2 * F))
        p_pos = np.clip(p_pos, 0.01, 0.99)  # 避免极端值
        p_neg = 1 - p_pos

        return np.vstack([p_neg, p_pos]).T

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.models:
            return {}

        importance = np.zeros(len(feature_names))
        total_alpha = 0

        for model, alpha in zip(self.models, self.alphas):
            if len(model.feature_importances_) == len(feature_names):
                importance += alpha * model.feature_importances_
                total_alpha += abs(alpha)

        if total_alpha > 0:
            importance = importance / total_alpha

        return dict(zip(feature_names, importance))