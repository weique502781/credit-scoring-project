import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib
from typing import Dict, List, Optional, Union

# 设置中文字体和负号显示
try:
    # Windows 系统
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    # Mac 系统
    # matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("✓ 中文字体设置成功")
except:
    print("⚠ 中文字体设置失败，可能显示方框")

# 核心修改1：固定项目根目录路径（修复路径拼接错误）
# 当前脚本路径：src/evaluation/visualizer.py
# 向上两级定位到项目根目录（credit-scoring-project）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

from src.evaluation.metrics import ModelMetrics  # 关联指标计算类


class ResultVisualizer:
    """模型结果可视化类，支持混淆矩阵、模型对比、特征重要性等可视化"""

    def __init__(self):
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 中文支持
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    def plot_confusion_matrices(
            self,
            confusion_matrices: Dict[str, List[List[int]]],
            save_path: str = "reports/confusion_matrices.png",
            figsize: tuple = (15, 8),
            normalize: bool = False
    ) -> None:
        """绘制多个模型的混淆矩阵（横向排列）"""
        n_models = len(confusion_matrices)
        if n_models == 0:
            raise ValueError("❌ 无混淆矩阵数据")

        # 核心修改2：拼接项目根目录路径
        save_path = os.path.join(PROJECT_ROOT, save_path)

        # 创建子图
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]  # 处理单模型情况

        # 绘制每个模型的混淆矩阵
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            cm = np.array(cm)
            # 归一化（按行）
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # 绘制热力图
            sns.heatmap(
                cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', ax=axes[idx], cbar=False,
                annot_kws={'fontsize': 10}
            )
            # 设置子图标题和标签
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('预测标签', fontsize=10)
            axes[idx].set_ylabel('真实标签', fontsize=10)
            axes[idx].set_xticklabels(['负类', '正类'], rotation=0)
            axes[idx].set_yticklabels(['负类', '正类'], rotation=0)

        # 添加总标题
        fig.suptitle('各模型混淆矩阵对比' + ('（归一化）' if normalize else ''),
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📥 混淆矩阵图已保存至: {save_path}")  # 输出保存路径

    def plot_model_comparison(
            self,
            metrics_df: pd.DataFrame,
            metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            save_path: str = "reports/model_comparison.png",
            figsize: tuple = (12, 8)
    ) -> None:
        """绘制多模型多指标对比柱状图"""
        # 验证输入
        required_cols = ['model'] + metrics
        if not all(col in metrics_df.columns for col in required_cols):
            missing = set(required_cols) - set(metrics_df.columns)
            raise ValueError(f"❌ DataFrame缺少列: {missing}")

        # 核心修改2：拼接项目根目录路径
        save_path = os.path.join(PROJECT_ROOT, save_path)

        # 数据重塑（长格式）
        metrics_long = pd.melt(
            metrics_df,
            id_vars=['model'],
            value_vars=metrics,
            var_name='metric',
            value_name='score'
        )
        # 绘制分组柱状图
        plt.figure(figsize=figsize)
        sns.barplot(
            x='model', y='score', hue='metric',
            data=metrics_long, palette='Set2', alpha=0.8
        )
        # 图表美化
        plt.title('多模型多指标对比', fontsize=14, fontweight='bold')
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('指标分数', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0.0, 1.05])
        plt.legend(title='评估指标', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        # 添加数值标签
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt='.3f', fontsize=8)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📥 模型对比图已保存至: {save_path}")  # 输出保存路径

    def plot_feature_importance(
            self,
            feature_importance: Dict[str, float],
            model_name: str,
            top_k: int = 10,
            save_path: str = "reports/feature_importance.png",
            figsize: tuple = (10, 8)
    ) -> None:
        """绘制特征重要性横向柱状图"""
        # 处理特征重要性数据
        if not feature_importance:
            raise ValueError("❌ 无特征重要性数据")

        # 核心修改2：拼接接项目根目录路径
        save_path = os.path.join(PROJECT_ROOT, save_path)

        # 排序并取top_k
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        features = [item[0] for item in sorted_importance]
        scores = [item[1] for item in sorted_importance]
        # 绘制横向柱状图
        plt.figure(figsize=figsize)
        colors = sns.color_palette('Blues_r', len(features))
        bars = plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
        # 图表美化
        plt.title(f'{model_name} 前{top_k}个重要特征', fontsize=14, fontweight='bold')
        plt.xlabel('特征重要性分数', fontsize=12)
        plt.ylabel('特征名称', fontsize=12)
        plt.yticks(range(len(features)), features)
        plt.grid(axis='x', alpha=0.3)
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{scores[i]:.3f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📥 特征重要性图已保存至: {save_path}")  # 输出保存路径

    def generate_summary_report(
            self,
            metrics_dict: Dict[str, Dict[str, Union[float, list]]],
            save_path: str = "reports/model_comparison.csv"
    ) -> pd.DataFrame:
        """生成模型评估汇总报告（CSV格式）"""
        # 核心修改2：拼接项目根目录路径
        save_path = os.path.join(PROJECT_ROOT, save_path)

        # 转换为DataFrame
        rows = []
        for model_name, metrics in metrics_dict.items():
            row = {'model': model_name}
            # 只保留数值型指标（排除混淆矩阵）
            for key, value in metrics.items():
                if key != 'confusion_matrix' and isinstance(value, (int, float)):
                    row[key] = round(value, 4)
            rows.append(row)
        metrics_df = pd.DataFrame(rows)
        # 保存为CSV
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
        metrics_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"📥 模型评估汇总报告已保存至: {save_path}")  # 输出保存路径
        return metrics_df


# 核心修改3：添加执行入口（触发输出和保存）
if __name__ == "__main__":
    # 1. 初始化可视化实例
    visualizer = ResultVisualizer()
    print("=" * 60)
    print("📊 开始模型结果可视化流程")
    print("=" * 60)

    # 2. 模拟输入数据（与 models 模块输出格式完全匹配）
    # 2.1 模拟混淆矩阵数据（6个模型，与 ensemble.py 中的模型列表一致）
    confusion_matrices = {
        "logistic_regression": [[142, 18], [25, 95]],  # 真实负类160，正类120
        "decision_tree": [[135, 25], [32, 88]],
        "svm_rbf": [[145, 15], [22, 98]],
        "naive_bayes": [[130, 30], [38, 82]],
        "custom_adaboost": [[150, 10], [18, 102]],
        "sklearn_adaboost": [[148, 12], [20, 100]]
    }

    # 2.2 模拟模型指标DataFrame（用于多指标对比）
    metrics_data = {
        "model": ["logistic_regression", "decision_tree", "svm_rbf", "naive_bayes", "custom_adaboost",
                  "sklearn_adaboost"],
        "accuracy": [0.835, 0.795, 0.845, 0.760, 0.860, 0.840],
        "precision": [0.838, 0.779, 0.868, 0.732, 0.912, 0.893],
        "recall": [0.792, 0.733, 0.817, 0.683, 0.850, 0.833],
        "f1": [0.814, 0.755, 0.842, 0.707, 0.880, 0.862],
        "roc_auc": [0.835, 0.798, 0.852, 0.762, 0.913, 0.887]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # 2.3 模拟特征重要性数据（以 custom_adaboost 为例）
    feature_importance = {
        "还款历史": 0.285,
        "负债比率": 0.213,
        "收入水平": 0.187,
        "信用年限": 0.125,
        "贷款金额": 0.098,
        "就业年限": 0.052,
        "家庭人数": 0.030,
        "住房类型": 0.010
    }

    # 2.4 模拟指标字典（用于生成CSV报告）
    metrics_dict = {
        "logistic_regression": {"accuracy": 0.835, "precision": 0.838, "recall": 0.792, "f1": 0.814, "roc_auc": 0.835},
        "custom_adaboost": {"accuracy": 0.860, "precision": 0.912, "recall": 0.850, "f1": 0.880, "roc_auc": 0.913}
    }

    # 3. 调用可视化方法（触发输出和图片保存）
    print("\n" + "-" * 60)
    print("1. 绘制混淆矩阵对比图")
    print("-" * 60)
    visualizer.plot_confusion_matrices(confusion_matrices)  # 原始混淆矩阵
    visualizer.plot_confusion_matrices(confusion_matrices, normalize=True,
                                       save_path="reports/confusion_matrices_normalized.png")  # 归一化混淆矩阵

    print("\n" + "-" * 60)
    print("2. 绘制多模型多指标对比图")
    print("-" * 60)
    visualizer.plot_model_comparison(metrics_df)

    print("\n" + "-" * 60)
    print("3. 绘制特征重要性图")
    print("-" * 60)
    visualizer.plot_feature_importance(feature_importance, model_name="custom_adaboost",
                                       save_path="reports/feature_importance_adaboost.png")

    print("\n" + "-" * 60)
    print("4. 生成模型评估CSV报告")
    print("-" * 60)
    visualizer.generate_summary_report(metrics_dict)

    print("\n" + "=" * 60)
    print("✅ 可视化流程完成！所有文件已保存至项目根目录的 reports 文件夹")
    print("=" * 60)