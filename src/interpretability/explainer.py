# ./interpretability/explainer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import warnings
import json
import os

warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Install with: pip install lime")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class ModelExplainer:
    """
    模型可解释性分析器，提供多种解释方法
    """

    def __init__(self, feature_names: List[str],
                 class_names: List[str] = None,
                 random_state: int = 42):
        """
        初始化解释器

        Args:
            feature_names: 特征名称列表
            class_names: 类别名称列表，默认为['Bad Credit', 'Good Credit']
            random_state: 随机种子
        """
        self.feature_names = feature_names

        if class_names is None:
            self.class_names = ['Bad Credit', 'Good Credit']
        else:
            self.class_names = class_names

        self.random_state = random_state
        self.lime_explainer = None
        self.shap_explainer = None
        self.project_root = self._get_project_root()

        # 设置可视化风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def _get_project_root(self) -> str:
        """获取项目根目录"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return project_root

    def initialize_lime_explainer(self, X_train: np.ndarray,
                                  categorical_features: List[int] = None):
        """
        初始化LIME解释器

        Args:
            X_train: 训练数据
            categorical_features: 分类特征的索引列表
        """
        if not LIME_AVAILABLE:
            print("LIME is not available. Please install it first.")
            return

        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                categorical_features=categorical_features,
                mode='classification',
                random_state=self.random_state,
                verbose=False
            )
            print("LIME解释器初始化成功")
        except Exception as e:
            print(f"LIME解释器初始化失败: {e}")

    def explain_with_lime(self, model, instance: np.ndarray,
                          num_features: int = 10,
                          show_plot: bool = True,
                          save_path: str = None) -> Dict:
        """
        使用LIME解释单个预测

        Args:
            model: 要解释的模型
            instance: 要解释的实例
            num_features: 要显示的特征数量
            show_plot: 是否显示可视化
            save_path: 结果保存路径

        Returns:
            解释结果字典
        """
        if self.lime_explainer is None:
            print("LIME解释器未初始化")
            return None

        try:
            # 生成解释
            exp = self.lime_explainer.explain_instance(
                data_row=instance,
                predict_fn=model.predict_proba,
                num_features=num_features,
                top_labels=1
            )

            # 提取解释信息
            explanation = {
                'prediction': model.predict(instance.reshape(1, -1))[0],
                'prediction_prob': model.predict_proba(instance.reshape(1, -1))[0],
                'explanation': exp.as_list(label=1),
                'local_pred': exp.local_pred,
                'intercept': exp.intercept[1] if hasattr(exp, 'intercept') else None
            }

            # 可视化
            if show_plot:
                self._plot_lime_explanation(exp, num_features, save_path)

            # 保存解释为HTML
            if save_path:
                html_path = save_path.replace('.png', '.html') if save_path else 'reports/lime_explanation.html'
                exp.save_to_file(html_path)
                print(f"LIME解释保存为HTML: {html_path}")

            return explanation

        except Exception as e:
            print(f"LIME解释失败: {e}")
            return None

    def _plot_lime_explanation(self, exp, num_features: int, save_path: str = None):
        """
        绘制LIME解释可视化

        Args:
            exp: LIME解释对象
            num_features: 显示的特征数量
            save_path: 保存路径
        """
        fig = exp.as_pyplot_figure(label=1)
        plt.title('LIME Feature Importance for Current Prediction', fontsize=14)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.tight_layout()

        # 创建报告目录
        reports_dir = os.path.join(self.project_root, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LIME可视化保存到: {save_path}")
        else:
            default_path = os.path.join(reports_dir, 'lime_explanation.png')
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"LIME可视化保存到: {default_path}")

        plt.show()

    def initialize_shap_explainer(self, model, X_train: np.ndarray,
                                  model_type: str = 'tree'):
        """
        初始化SHAP解释器

        Args:
            model: 要解释的模型
            X_train: 训练数据
            model_type: 模型类型 ('tree', 'linear', 'kernel')
        """
        if not SHAP_AVAILABLE:
            print("SHAP is not available. Please install it first.")
            return

        try:
            if model_type == 'tree' and hasattr(model, 'tree_'):
                self.shap_explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                self.shap_explainer = shap.LinearExplainer(model, X_train)
            else:
                # 使用KernelExplainer（通用但较慢）
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba,
                    shap.sample(X_train, 100)  # 使用样本加速
                )

            print("SHAP解释器初始化成功")

        except Exception as e:
            print(f"SHAP解释器初始化失败: {e}")

    def explain_with_shap(self, model, X: np.ndarray,
                          max_samples: int = 100,
                          save_path: str = None) -> Tuple:
        """
        使用SHAP解释模型

        Args:
            model: 要解释的模型
            X: 要解释的数据
            max_samples: 最大样本数（用于加速）
            save_path: 结果保存路径前缀

        Returns:
            (shap_values, shap_explainer) 元组
        """
        if self.shap_explainer is None:
            print("SHAP解释器未初始化")
            return None, None

        try:
            # 限制样本数
            if len(X) > max_samples:
                X_sample = X[:max_samples]
            else:
                X_sample = X

            # 计算SHAP值
            shap_values = self.shap_explainer.shap_values(X_sample)

            # 创建可视化
            self._create_shap_visualizations(shap_values, X_sample, save_path)

            return shap_values, self.shap_explainer

        except Exception as e:
            print(f"SHAP解释失败: {e}")
            return None, None

    def _create_shap_visualizations(self, shap_values, X, save_path: str = None):
        """
        创建SHAP可视化

        Args:
            shap_values: SHAP值
            X: 特征数据
            save_path: 保存路径前缀
        """
        try:
            # 创建报告目录
            reports_dir = os.path.join(self.project_root, 'reports')
            os.makedirs(reports_dir, exist_ok=True)

            # 确定保存路径
            if save_path:
                base_path = save_path.replace('.png', '')
            else:
                base_path = os.path.join(reports_dir, 'shap')

            # 1. 汇总图
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list):
                # 多分类
                shap.summary_plot(shap_values[1], X,
                                  feature_names=self.feature_names[:X.shape[1]],
                                  show=False)
            else:
                # 二分类
                shap.summary_plot(shap_values, X,
                                  feature_names=self.feature_names[:X.shape[1]],
                                  show=False)

            plt.title('SHAP Feature Importance Summary', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{base_path}_summary.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 2. 条形图
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X,
                                  feature_names=self.feature_names[:X.shape[1]],
                                  plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values, X,
                                  feature_names=self.feature_names[:X.shape[1]],
                                  plot_type="bar", show=False)

            plt.title('SHAP Feature Importance (Bar)', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{base_path}_bar.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"SHAP可视化已保存到: {base_path}_*.png")

        except Exception as e:
            print(f"SHAP可视化创建失败: {e}")

    def analyze_feature_importance(self, model) -> Dict:
        """
        分析特征重要性

        Args:
            model: 模型

        Returns:
            特征重要性字典
        """
        importance = {}

        # 方法1: 模型内置特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, feat in enumerate(self.feature_names[:len(importances)]):
                importance[feat] = importances[i]

        # 方法2: 系数大小（线性模型）
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            for i, feat in enumerate(self.feature_names[:len(coef)]):
                importance[feat] = abs(coef[i])

        # 排序并归一化
        if importance:
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            total = sum([v for _, v in sorted_importance])
            normalized = {k: v / total for k, v in sorted_importance}

            # 可视化
            self._plot_feature_importance(normalized)

            return normalized

        return {}

    def _plot_feature_importance(self, importance_dict: Dict):
        """
        绘制特征重要性图

        Args:
            importance_dict: 特征重要性字典
        """
        if not importance_dict:
            return

        # 取前15个特征
        top_features = dict(sorted(importance_dict.items(),
                                   key=lambda x: x[1],
                                   reverse=True)[:15])

        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        values = list(top_features.values())

        # 创建水平条形图
        y_pos = np.arange(len(features))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, features)
        plt.xlabel('Importance Score')
        plt.title('Top 15 Feature Importance', fontsize=14)
        plt.tight_layout()

        # 保存到报告目录
        reports_dir = os.path.join(self.project_root, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, 'feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图保存到: {save_path}")
        plt.show()

    def create_explanation_report(self, model, X: np.ndarray,
                                  sample_indices: List[int] = None,
                                  save_dir: str = None) -> Dict:
        """
        创建完整的解释报告

        Args:
            model: 要解释的模型
            X: 特征数据
            sample_indices: 要解释的样本索引列表
            save_dir: 报告保存目录

        Returns:
            解释报告字典
        """
        if save_dir is None:
            save_dir = os.path.join(self.project_root, 'reports', 'explanations')
        os.makedirs(save_dir, exist_ok=True)

        report = {
            'model_type': type(model).__name__,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'explanations': []
        }

        # 全局解释
        report['global_explanation'] = self.analyze_feature_importance(model)

        # 样本解释
        if sample_indices is None:
            # 默认选择5个代表性样本
            sample_indices = list(range(min(5, len(X))))

        for idx in sample_indices:
            if idx < len(X):
                sample = X[idx]

                # LIME解释
                lime_result = None
                if LIME_AVAILABLE and self.lime_explainer:
                    lime_result = self.explain_with_lime(
                        model, sample,
                        num_features=8,
                        show_plot=False,
                        save_path=os.path.join(save_dir, f'sample_{idx}_lime.png')
                    )

                sample_explanation = {
                    'sample_index': idx,
                    'prediction': int(model.predict(sample.reshape(1, -1))[0]),
                    'prediction_prob': model.predict_proba(sample.reshape(1, -1))[0].tolist(),
                    'lime_explanation': lime_result
                }
                report['explanations'].append(sample_explanation)

        # 保存报告
        report_path = os.path.join(save_dir, 'explanation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"解释报告保存到: {report_path}")
        return report


# 使用示例
if __name__ == "__main__":
    # 示例用法
    print("模型可解释性分析器")
    print("=" * 50)

    # 创建示例特征名称
    feature_names = [f'feature_{i}' for i in range(10)]

    # 初始化解释器
    explainer = ModelExplainer(
        feature_names=feature_names,
        class_names=['Good Credit', 'Bad Credit']
    )

    print(f"解释器初始化完成")
    print(f"特征数量: {len(feature_names)}")
    print(f"LIME可用: {LIME_AVAILABLE}")
    print(f"SHAP可用: {SHAP_AVAILABLE}")