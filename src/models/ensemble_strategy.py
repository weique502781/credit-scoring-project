from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
import joblib


class EnsembleVoting:
    """
    多种集成策略：
    1. 硬投票
    2. 软投票（平均概率）
    3. 加权投票
    4. Stacking（可选）
    """

    def __init__(self):
        self.models: Dict[str, object] = {}

    def load_models(self, model_dir: str):
        """从磁盘加载所有模型"""
        import os
        import glob

        model_files = glob.glob(f"{model_dir}/*.pkl")

        for file_path in model_files:
            model_name = os.path.basename(file_path).replace('.pkl', '')
            try:
                model = joblib.load(file_path)
                self.models[model_name] = model
                print(f"[Voting] 已加载模型: {model_name}")
            except Exception as e:
                print(f"[Warning] 加载模型 {model_name} 失败: {e}")

    def hard_voting(self, X: np.ndarray) -> np.ndarray:
        """硬投票（多数表决）"""
        if not self.models:
            raise ValueError("请先加载模型")

        all_predictions = []
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                all_predictions.append(pred)

        all_predictions = np.array(all_predictions)  # (n_models, n_samples)

        # 多数投票
        from scipy import stats
        return stats.mode(all_predictions, axis=0)[0].flatten()

    def soft_voting(self, X: np.ndarray) -> np.ndarray:
        """软投票（平均概率）"""
        if not self.models:
            raise ValueError("请先加载模型")

        all_probas = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba)

        # 平均概率
        avg_proba = np.mean(all_probas, axis=0)
        return np.argmax(avg_proba, axis=1)

    def weighted_voting(self, X: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """加权投票"""
        if not self.models:
            raise ValueError("请先加载模型")

        weighted_sum = np.zeros((X.shape[0], 2))

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                weight = weights.get(name, 1.0)
                weighted_sum += weight * proba

        return np.argmax(weighted_sum, axis=1)

    def get_best_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                           model_name: str = "adaboost_custom") -> float:
        """找到最佳分类阈值（用于调整分类边界）"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")

        model = self.models[model_name]
        if not hasattr(model, 'predict_proba'):
            return 0.5

        # 预测概率
        y_proba = model.predict_proba(X_val)[:, 1]

        # 寻找最佳阈值（最大化F1分数）
        from sklearn.metrics import f1_score

        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.3, 0.7, 0.02):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"[Threshold] 模型 {model_name} 最佳阈值: {best_threshold:.3f}, F1: {best_f1:.3f}")
        return best_threshold