from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
import joblib
from sklearn.metrics import roc_auc_score, f1_score

def compute_model_weights(
    models: Dict[str, object],
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "auc_f1",
    alpha: float = 0.7
) -> Dict[str, float]:
    """
    基于验证集性能自动计算集成权重（唯一方案）

    权重 = alpha * AUC + (1 - alpha) * F1
    """

    raw_scores = {}

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue

        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, (y_prob >= 0.5).astype(int))

        if method == "auc_f1":
            score = alpha * auc + (1 - alpha) * f1
        elif method == "auc":
            score = auc
        else:
            raise ValueError(f"Unknown weight method: {method}")

        raw_scores[name] = score

    total = sum(raw_scores.values())
    if total == 0:
        raise ValueError("模型权重计算失败：得分全为 0")

    weights = {k: v / total for k, v in raw_scores.items()}

    print("[Ensemble Weights | AUC + F1]")
    for k, v in weights.items():
        print(f"  {k}: {v:.3f}")

    return weights


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

    def weighted_voting_proba(self, X: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """加权投票：返回融合后的概率 (n_samples, 2)"""
        if not self.models:
            raise ValueError("请先加载模型")

        weighted_sum = np.zeros((X.shape[0], 2))
        total_weight = 0.0

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                w = weights.get(name, 1.0)
                weighted_sum += w * proba
                total_weight += w

        if total_weight <= 0:
            raise ValueError("weights 总和必须 > 0")

        return weighted_sum / total_weight

    def apply_threshold(self, y_proba: np.ndarray, threshold: float) -> np.ndarray:
        """用阈值把概率转成最终 0/1"""
        return (y_proba[:, 1] >= threshold).astype(int)


class FinalEnsembleModel:
    """最终可落盘的集成模型：融合概率 + 阈值决策"""

    def __init__(self, models: Dict[str, object], weights: Dict[str, float], threshold: float = 0.5):
        self.models = models
        self.weights = weights
        self.threshold = threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        weighted_sum = np.zeros((X.shape[0], 2))
        total_weight = 0.0

        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                w = self.weights.get(name, 1.0)
                weighted_sum += w * proba
                total_weight += w

        if total_weight <= 0:
            raise ValueError("weights 总和必须 > 0")

        return weighted_sum / total_weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba_pos = self.predict_proba(X)[:, 1]
        return (proba_pos >= self.threshold).astype(int)
