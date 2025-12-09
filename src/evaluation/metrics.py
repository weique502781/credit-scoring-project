import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Union

class ModelMetrics:
    """模型评估指标计算类，支持二分类任务的全面指标评估"""
    @staticmethod
    def calculate_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: np.ndarray,
            average: str = 'binary',
            pos_label: int = 1
    ) -> Dict[str, Union[float, list]]:
        # 原有方法实现（不变）
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
            ),
            'recall': recall_score(
                y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
            ),
            'f1': f1_score(
                y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
            ),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, Union[float, list]], model_name: str) -> None:
        # 原有方法实现（不变）
        print("=" * 60)
        print(f"📊 {model_name} 评估指标详情")
        print("=" * 60)
        print(f"准确率 (Accuracy):    {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision):   {metrics['precision']:.4f}")
        print(f"召回率 (Recall):      {metrics['recall']:.4f}")
        print(f"F1分数 (F1-Score):    {metrics['f1']:.4f}")
        print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
        print("\n混淆矩阵 (Confusion Matrix):")
        cm = np.array(metrics['confusion_matrix'])
        print(f"          预测负类    预测正类")
        print(f"真实负类    {cm[0, 0]:<8} {cm[0, 1]:<8}")
        print(f"真实正类    {cm[1, 0]:<8} {cm[1, 1]:<8}")
        print("=" * 60 + "\n")

# -------------------------- 新增：执行入口代码 --------------------------
if __name__ == "__main__":
    # 1. 模拟测试数据（替换为你的真实数据，格式需匹配二分类任务）
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 真实标签（0/1）
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0])  # 模型预测的类别标签
    y_prob = np.array([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7],
                       [0.4, 0.6], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])  # 模型预测的概率（二维数组）

    # 2. 调用方法计算指标
    metrics_result = ModelMetrics.calculate_metrics(y_true, y_pred, y_prob)

    # 3. 调用方法打印结果
    ModelMetrics.print_metrics(metrics_result, model_name="测试模型（二分类）")