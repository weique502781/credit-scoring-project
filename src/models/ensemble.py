# src/models/ensemble.py

import numpy as np
from typing import Dict
from sklearn.ensemble import AdaBoostClassifier

# 导入优化后的基础模型
from .base_models import (
    LogisticRegressionModel,
    DecisionTreeModel,
    SVMModel,
    NaiveBayesModel,
)

from .adaboost_custom import ImprovedAdaBoost


class EnsembleTrainer:
    """
    负责同时训练多种模型，便于比较性能、统一管理。
    """

    def __init__(self, random_state: int = 42):
        """
            random_state: 随机种子
        """
        # 使用优化配置（根据测试结果最佳）
        self.models: Dict[str, object] = {
            "logistic_regression": LogisticRegressionModel(),
            "decision_tree": DecisionTreeModel(),
            "svm_rbf": SVMModel(),
            "naive_bayes": NaiveBayesModel(),
            "adaboost_custom": ImprovedAdaBoost(
                n_estimators=100,
                learning_rate=0.8,
                weak_learner_depth=3,
                random_state=random_state
            ),
            "sklearn_adaboost": AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=random_state
            )
        }

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练所有模型"""
        for name, model in self.models.items():
            print(f"[Ensemble] 训练模型：{name}")
            if hasattr(model, 'train'):
                model.train(X_train, y_train)
            else:
                # sklearn AdaBoost 直接fit
                model.fit(X_train, y_train)

    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """使用所有模型进行预测"""
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        return preds

    def predict_proba_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """使用所有模型进行概率预测"""
        probas = {}
        for name, model in self.models.items():
            probas[name] = model.predict_proba(X)
        return probas

    def get_models(self) -> Dict[str, object]:
        """返回训练好的模型字典"""
        return self.models

    def quick_evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """快速评估所有模型"""
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            results[name] = accuracy
            print(f"{name}: accuracy = {accuracy:.4f}")
        return results