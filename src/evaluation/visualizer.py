import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


class ResultVisualizer:
    """模型结果可视化类"""

    @staticmethod
    def plot_confusion_matrices(confusion_matrices: dict, save_path: str = None) -> None:
        """绘制多个模型的混淆矩阵"""
        n_models = len(confusion_matrices)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for i, (model_name, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} 混淆矩阵')
            axes[i].set_xlabel('预测标签')
            axes[i].set_ylabel('真实标签')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵图已保存至: {save_path}")

        plt.close()

    @staticmethod
    def plot_model_comparison(metrics_df: pd.DataFrame, metric: str = 'roc_auc',
                              save_path: str = None) -> None:
        """绘制模型对比柱状图"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y=metric, data=metrics_df)
        plt.title(f'模型{metric}对比')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型对比图已保存至: {save_path}")

        plt.close()

    @staticmethod
    def plot_feature_importance(importance: np.ndarray, feature_names: list,
                                model_name: str, save_path: str = None) -> None:
        """绘制特征重要性图"""
        plt.figure(figsize=(10, 8))
        indices = np.argsort(importance)[::-1]
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title(f'{model_name} 特征重要性')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存至: {save_path}")

        plt.close()