import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib

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
            # 确保cm是numpy数组
            if isinstance(cm, list):
                cm = np.array(cm)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} 混淆矩阵', fontsize=14)
            axes[i].set_xlabel('预测标签', fontsize=12)
            axes[i].set_ylabel('真实标签', fontsize=12)

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
        plt.figure(figsize=(12, 8))

        # 确保有数据
        if metrics_df.empty:
            print(f"⚠ 没有数据用于绘制 {metric} 对比图")
            plt.close()
            return

        # 简化和格式化模型名称
        def format_model_name(name):
            name_mapping = {
                'logistic_regression': '逻辑回归',
                'decision_tree': '决策树',
                'svm_rbf': 'SVM',
                'naive_bayes': '朴素贝叶斯',
                'adaboost_custom': '自定义AdaBoost',
                'sklearn_adaboost': 'Sklearn AdaBoost'
            }
            return name_mapping.get(name, name)

        metrics_df['model_formatted'] = metrics_df['model'].apply(format_model_name)

        # 绘制条形图
        ax = sns.barplot(x='model_formatted', y=metric, data=metrics_df)
        plt.title(f'模型{metric}对比', fontsize=16)
        plt.xlabel('模型', fontsize=14)
        plt.ylabel(metric, fontsize=14)

        # 在条形上添加数值
        for i, v in enumerate(metrics_df[metric]):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=12)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型对比图已保存至: {save_path}")

        plt.close()

    @staticmethod
    def plot_feature_importance(importance: np.ndarray, feature_names: list,
                                model_name: str, save_path: str = None,
                                top_n: int = 15) -> None:
        """绘制特征重要性图 - 修复中文显示问题"""
        plt.figure(figsize=(14, 10))

        # 确保importance是numpy数组
        if isinstance(importance, list):
            importance = np.array(importance)

        # 如果特征名是原始的特征名（可能包含feature_前缀），进行格式化
        formatted_names = []
        for name in feature_names:
            # 移除feature_前缀并格式化
            if isinstance(name, str):
                if name.startswith('feature_'):
                    # 尝试提取数字并格式化
                    try:
                        num = int(name.replace('feature_', ''))
                        formatted_names.append(f'特征{num}')
                    except:
                        formatted_names.append(name)
                else:
                    # 如果是分类特征编码后的名称，进行简化
                    if '_' in name and name.split('_')[0] in feature_names:
                        base_name = name.split('_')[0]
                        formatted_names.append(f'{base_name}类别')
                    else:
                        formatted_names.append(name)
            else:
                formatted_names.append(str(name))

        # 按重要性排序，取前top_n个
        indices = np.argsort(importance)[::-1][:top_n]
        sorted_importance = importance[indices]
        sorted_names = [formatted_names[i] for i in indices]

        # 创建水平条形图
        y_pos = np.arange(len(sorted_names))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))

        bars = plt.barh(y_pos, sorted_importance, color=colors, edgecolor='black')
        plt.yticks(y_pos, sorted_names, fontsize=12)

        # 添加数值标签
        for i, v in enumerate(sorted_importance):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10)

        plt.xlabel('特征重要性', fontsize=14)
        plt.title(f'{model_name} 特征重要性 (Top {top_n})', fontsize=16, pad=20)

        # 添加网格
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 尝试不同的保存选项
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ 特征重要性图已保存至: {save_path}")
            except Exception as e:
                print(f"✗ 保存图片失败: {e}")
                # 尝试不使用中文字体保存
                original_font = matplotlib.rcParams['font.sans-serif']
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.savefig(save_path.replace('.png', '_en.png'), dpi=300, bbox_inches='tight')
                matplotlib.rcParams['font.sans-serif'] = original_font
                print(f"✓ 使用英文字体保存为: {save_path.replace('.png', '_en.png')}")

        plt.close()

    @staticmethod
    def plot_roc_curves_single(fpr: dict, tpr: dict, auc_values: dict,
                               save_path: str = None) -> None:
        """绘制单个ROC曲线图"""
        plt.figure(figsize=(12, 10))

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')

        # 绘制各模型ROC曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(fpr)))
        for i, (model_name, fpr_values) in enumerate(fpr.items()):
            plt.plot(fpr_values, tpr[model_name], lw=3, color=colors[i],
                     label=f'{model_name} (AUC = {auc_values[model_name]:.3f})')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('假正例率 (FPR)', fontsize=14)
        plt.ylabel('真正例率 (TPR)', fontsize=14)
        plt.title('ROC曲线对比', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线图已保存至: {save_path}")

        plt.close()

    @staticmethod
    def create_summary_report(metrics_df: pd.DataFrame, save_path: str = None) -> None:
        """创建模型性能总结报告"""
        if metrics_df.empty:
            print("⚠ 没有数据用于创建总结报告")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes) and metric in metrics_df.columns:
                ax = axes[idx]

                # 简化和格式化模型名称
                def format_model_name(name):
                    name_mapping = {
                        'logistic_regression': '逻辑回归',
                        'decision_tree': '决策树',
                        'svm_rbf': 'SVM',
                        'naive_bayes': '朴素贝叶斯',
                        'adaboost_custom': '自定义AdaBoost',
                        'sklearn_adaboost': 'Sklearn AdaBoost'
                    }
                    return name_mapping.get(name, name)

                temp_df = metrics_df.copy()
                temp_df['model'] = temp_df['model'].apply(format_model_name)

                bars = ax.bar(range(len(temp_df)), temp_df[metric])
                ax.set_title(f'{metric}对比', fontsize=14)
                ax.set_xticks(range(len(temp_df)))
                ax.set_xticklabels(temp_df['model'], rotation=45, ha='right')
                ax.set_ylim([0, 1.1])

                # 添加数值标签
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        # 隐藏多余的子图
        for idx in range(len(metrics_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('模型性能总结', fontsize=18, y=1.02)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"总结报告图已保存至: {save_path}")

        plt.close()