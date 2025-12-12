import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score
)
from typing import Dict, Optional, List
import os
from src.models.ensemble import EnsembleTrainer  # 关联集成模型类
import sys
# 向上两级找到项目根目录（因为当前脚本在 src/evaluation 下）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


class ROCAnalyzer:
    """ROC曲线分析类，支持多模型对比、阈值优化和可视化保存"""
    def __init__(self):
        self.models_roc_data: Dict[str, Dict[str, np.ndarray]] = {}  # 存储模型ROC数据
        self.models_pr_data: Dict[str, Dict[str, np.ndarray]] = {}  # 存储精确率-召回率数据

    def add_model(
            self,
            model_name: str,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            pos_label: int = 1
    ) -> None:
        """
        添加模型的预测结果用于ROC和PR曲线分析
        Args:
            model_name: 模型名称（唯一标识）
            y_true: 真实标签
            y_prob: 预测概率（二维数组：[n_samples, 2]）
            pos_label: 正类标签（默认1）
        """
        # 提取正类概率
        y_pos_prob = y_prob[:, pos_label]
        # 计算ROC曲线数据
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pos_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        self.models_roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds,
            'auc': roc_auc
        }
        # 计算精确率-召回率曲线数据
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pos_prob, pos_label=pos_label)
        pr_auc = auc(recall, precision)
        self.models_pr_data[model_name] = {
            'precision': precision,
            'recall': recall,
            'thresholds': pr_thresholds,
            'auc': pr_auc
        }
        print(f"✅ 已添加 {model_name} 分析数据 | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    def plot_roc_curves(
            self,
            save_path: str = "reports/roc_curves.png",
            title: str = "多模型ROC曲线对比",
            figsize: tuple = (10, 8)
    ) -> None:
        save_path = os.path.join(PROJECT_ROOT, save_path)
        """
        绘制所有模型的ROC曲线（含随机猜测基准线）
        Args:
            save_path: 图片保存路径（默认保存到reports目录）
            title: 图表标题
            figsize: 图表尺寸
        """
        if not self.models_roc_data:
            raise ValueError("❌ 无模型数据，请先调用add_model添加数据")
        # 创建图表
        plt.figure(figsize=figsize)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 支持中文
        # 绘制随机猜测基准线（AUC=0.5）
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC=0.5)')
        # 绘制各模型ROC曲线
        for model_name, data in self.models_roc_data.items():
            plt.plot(
                data['fpr'], data['tpr'],
                lw=3, alpha=0.8,
                label=f'{model_name} (AUC={data["auc"]:.3f})'
            )
        # 图表美化
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('假正例率 (FPR)', fontsize=12)
        plt.ylabel('真正例率 (TPR)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📥 ROC曲线已保存至: {save_path}")

    def plot_pr_curves(
            self,
            save_path: str = "reports/pr_curves.png",
            title: str = "多模型精确率-召回率曲线对比",
            figsize: tuple = (10, 8)
    ) -> None:
        save_path = os.path.join(PROJECT_ROOT, save_path)
        """绘制所有模型的精确率-召回率曲线"""
        if not self.models_pr_data:
            raise ValueError("❌ 无模型数据，请先调用add_model添加数据")
        plt.figure(figsize=figsize)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        for model_name, data in self.models_pr_data.items():
            plt.plot(
                data['recall'], data['precision'],
                lw=3, alpha=0.8,
                label=f'{model_name} (PR-AUC={data["auc"]:.3f})'
            )
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📥 PR曲线已保存至: {save_path}")

    def find_best_threshold(
            self,
            model_name: str,
            criterion: str = "f1",
            y_true: Optional[np.ndarray] = None,
            y_prob: Optional[np.ndarray] = None
    ) -> float:
        """
        寻找最佳分类阈值（支持F1分数、Youden指数两种准则）
        Args:
            model_name: 目标模型名称
            criterion: 优化准则（"f1" 或 "youden"）
            y_true: 真实标签（若未添加模型数据需传入）
            y_prob: 预测概率（若未添加模型数据需传入）
        Returns:
            最佳阈值
        """
        # 若未添加模型数据，先临时添加
        if model_name not in self.models_roc_data and y_true is not None and y_prob is not None:
            self.add_model(model_name, y_true, y_prob)
        if model_name not in self.models_roc_data:
            raise ValueError(f"❌ 模型 {model_name} 未找到，请先添加数据")
        roc_data = self.models_roc_data[model_name]
        pr_data = self.models_pr_data[model_name]
        if criterion == "f1":
            # 基于精确率-召回率曲线优化F1分数
            precision, recall, thresholds = pr_data['precision'], pr_data['recall'], pr_data['thresholds']
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_score = f1_scores[best_idx]
            print(f"🎯 {model_name} 最佳F1阈值: {best_threshold:.3f} (F1={best_score:.4f})")
        elif criterion == "youden":
            # 基于ROC曲线优化Youden指数（TPR - FPR）
            youden_indices = roc_data['tpr'] - roc_data['fpr']
            best_idx = np.argmax(youden_indices)
            best_threshold = roc_data['thresholds'][best_idx]
            best_score = youden_indices[best_idx]
            print(f"🎯 {model_name} 最佳Youden阈值: {best_threshold:.3f} (指数={best_score:.4f})")
        else:
            raise ValueError(f"❌ 不支持的准则 {criterion}，可选：'f1'、'youden'")
        return best_threshold

    def threshold_analysis(
            self,
            model_name: str,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            save_path: str = "reports/threshold_analysis.png",
            thresholds: Optional[np.ndarray] = None
    ) -> None:
        save_path = os.path.join(PROJECT_ROOT, save_path)
        """
        分析不同阈值对精确率、召回率、F1分数的影响
        Args:
            model_name: 模型名称
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 图片保存路径
            thresholds: 自定义阈值范围（默认0.05~0.95，步长0.02）
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.95, 0.02)
        # 提取正类概率
        y_pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        # 计算各阈值下的指标
        precisions, recalls, f1s = [], [], []
        for thres in thresholds:
            y_pred = (y_pos_prob >= thres).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        # 绘制阈值分析图
        plt.figure(figsize=(12, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.plot(thresholds, precisions, 'b-', lw=3, label='精确率', marker='o', markersize=4)
        plt.plot(thresholds, recalls, 'g-', lw=3, label='召回率', marker='s', markersize=4)
        plt.plot(thresholds, f1s, 'r-', lw=3, label='F1分数', marker='^', markersize=4)
        # 标记最佳F1阈值
        best_f1_idx = np.argmax(f1s)
        best_thres = thresholds[best_f1_idx]
        plt.axvline(x=best_thres, color='orange', linestyle='--', lw=2,
                    label=f'最佳阈值: {best_thres:.3f}')
        # 图表美化
        plt.xlabel('分类阈值', fontsize=12)
        plt.ylabel('指标分数', fontsize=12)
        plt.title(f'{model_name} 阈值敏感性分析', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📥 阈值分析图已保存至: {save_path}")

# -------------------------- 执行入口代码（新增核心部分） --------------------------
if __name__ == "__main__":
    # 1. 模拟测试数据（严格匹配 models 模块输出格式）
    np.random.seed(42)  # 固定随机种子，结果可复现
    n_samples = 200  # 模拟200个样本
    y_true = np.random.randint(0, 2, size=n_samples)  # 真实标签（0/1二分类）

    # 模拟3个模型的预测概率（均为 [n_samples, 2] 格式，与 models 输出一致）
    # 模型1：custom_adaboost
    y_prob_adaboost = np.random.rand(n_samples, 2)
    y_prob_adaboost = y_prob_adaboost / y_prob_adaboost.sum(axis=1, keepdims=True)  # 概率归一化
    # 模型2：logistic_regression
    y_prob_lr = np.random.rand(n_samples, 2)
    y_prob_lr = y_prob_lr / y_prob_lr.sum(axis=1, keepdims=True)
    # 模型3：svm_rbf
    y_prob_svm = np.random.rand(n_samples, 2)
    y_prob_svm = y_prob_svm / y_prob_svm.sum(axis=1, keepdims=True)

    # 2. 创建分析实例并执行核心流程
    roc_analyzer = ROCAnalyzer()
    print("=" * 60)
    print("📊 开始多模型ROC/PR曲线分析流程")
    print("=" * 60)

    # 3. 添加所有模型数据（触发控制台输出）
    roc_analyzer.add_model("custom_adaboost", y_true, y_prob_adaboost)
    roc_analyzer.add_model("logistic_regression", y_true, y_prob_lr)
    roc_analyzer.add_model("svm_rbf", y_true, y_prob_svm)

    # 4. 绘制并保存曲线（自动创建 reports 目录）
    print("\n" + "-" * 60)
    print("📈 开始绘制曲线并保存")
    print("-" * 60)
    roc_analyzer.plot_roc_curves()  # 保存ROC曲线到 reports/roc_curves.png
    roc_analyzer.plot_pr_curves()   # 保存PR曲线到 reports/pr_curves.png

    # 5. 阈值分析（以 custom_adaboost 为例）
    print("\n" + "-" * 60)
    print("🎯 开始阈值敏感性分析")
    print("-" * 60)
    roc_analyzer.threshold_analysis(
        model_name="custom_adaboost",
        y_true=y_true,
        y_prob=y_prob_adaboost,
        save_path="reports/threshold_analysis_adaboost.png"
    )

    # 6. 寻找最佳阈值（F1准则）
    print("\n" + "-" * 60)
    print("🔍 寻找最佳分类阈值")
    print("-" * 60)
    roc_analyzer.find_best_threshold(
        model_name="custom_adaboost",
        criterion="f1",
        y_true=y_true,
        y_prob=y_prob_adaboost
    )
    roc_analyzer.find_best_threshold(
        model_name="logistic_regression",
        criterion="f1",
        y_true=y_true,
        y_prob=y_prob_lr
    )

    print("\n" + "=" * 60)
    print("✅ 所有分析流程完成！结果已保存至 reports 目录")
    print("=" * 60)