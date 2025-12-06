import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os


class ROCAnalyzer:
    """ROC曲线分析与绘制类"""

    def __init__(self):
        self.models_data = {}  # 存储模型的fpr、tpr和auc

    def add_model(self, model_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """添加模型的预测概率用于ROC分析"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        self.models_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
        print(f"已添加 {model_name} 的ROC数据，AUC: {roc_auc:.4f}")

    def plot_roc_curves(self, save_path: str = None, title: str = "ROC曲线对比") -> None:
        """绘制所有模型的ROC曲线"""
        plt.figure(figsize=(10, 8))

        # 绘制对角线（随机猜测基准）
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        # 绘制各模型ROC曲线
        for name, data in self.models_data.items():
            plt.plot(data['fpr'], data['tpr'], lw=2,
                     label=f'{name} (AUC = {data["auc"]:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title(title)
        plt.legend(loc="lower right")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存至: {save_path}")

        plt.close()

    def threshold_analysis(self, model_name: str, y_true: np.ndarray, y_prob: np.ndarray,
                           save_path: str = None) -> None:
        """分析不同阈值对精确率和召回率的影响"""
        if model_name not in self.models_data:
            raise ValueError(f"模型 {model_name} 未添加，请先调用add_model方法")

        thresholds = np.arange(0.1, 1.0, 0.05)
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, 'b-', label='精确率')
        plt.plot(thresholds, recalls, 'g-', label='召回率')
        plt.xlabel('分类阈值')
        plt.ylabel('分数')
        plt.title(f'{model_name} 阈值分析')
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"阈值分析图已保存至: {save_path}")

        plt.close()