import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import warnings

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

        # 设置可视化风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

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

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LIME可视化保存到: {save_path}")
        else:
            plt.savefig('reports/lime_explanation.png', dpi=300, bbox_inches='tight')
            print("LIME可视化保存到: reports/lime_explanation.png")

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
            # 确定保存路径
            if save_path:
                base_path = save_path.replace('.png', '')
            else:
                base_path = 'reports/shap'

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

            # 3. 依赖图（前5个重要特征）
            if hasattr(self.shap_explainer, 'expected_value'):
                expected_value = self.shap_explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[1]  # 二分类取正类

                # 计算特征重要性
                feature_importance = np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(0)
                top_features = np.argsort(feature_importance)[-5:][::-1]

                for i, feature_idx in enumerate(top_features):
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(feature_idx,
                                         shap_values[1] if isinstance(shap_values, list) else shap_values,
                                         X,
                                         feature_names=self.feature_names[:X.shape[1]],
                                         show=False)
                    plt.title(f'SHAP Dependence Plot: {self.feature_names[feature_idx]}', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f'{base_path}_dependence_{i + 1}.png', dpi=300, bbox_inches='tight')
                    plt.show()

            print(f"SHAP可视化已保存到: {base_path}_*.png")

        except Exception as e:
            print(f"SHAP可视化创建失败: {e}")

    def create_global_explanation(self, model, X_train: np.ndarray,
                                  method: str = 'both',
                                  save_path: str = None) -> Dict:
        """
        创建全局解释

        Args:
            model: 要解释的模型
            X_train: 训练数据
            method: 解释方法 ('lime', 'shap', 'both')
            save_path: 保存路径前缀

        Returns:
            包含全局解释的字典
        """
        explanation = {}

        # 特征重要性分析
        explanation['feature_importance'] = self._analyze_feature_importance(model)

        # 决策边界分析（仅对简单模型）
        if X_train.shape[1] <= 10:  # 特征维度较低时
            explanation['decision_boundary'] = self._analyze_decision_boundary(model, X_train)

        # 部分依赖分析
        explanation['partial_dependence'] = self._analyze_partial_dependence(model, X_train)

        # 选择的方法
        if method in ['lime', 'both'] and LIME_AVAILABLE:
            # 使用LIME解释多个代表性样本
            explanation['lime_global'] = self._create_lime_global_explanation(model, X_train)

        if method in ['shap', 'both'] and SHAP_AVAILABLE:
            # 使用SHAP解释
            self.initialize_shap_explainer(model, X_train)
            shap_values, _ = self.explain_with_shap(model, X_train[:100],
                                                    save_path=save_path)
            explanation['shap_values'] = shap_values

        # 保存全局解释
        if save_path:
            self._save_global_explanation(explanation, save_path)

        return explanation

    def _analyze_feature_importance(self, model) -> Dict:
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
        plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_decision_boundary(self, model, X_train: np.ndarray) -> Dict:
        """
        分析决策边界（仅适用于2-3个特征）

        Args:
            model: 模型
            X_train: 训练数据

        Returns:
            决策边界分析结果
        """
        # 这里只是示例，实际实现需要根据特征维度调整
        return {
            'note': 'Decision boundary analysis requires feature selection and visualization',
            'available': X_train.shape[1] <= 3
        }

    def _analyze_partial_dependence(self, model, X_train: np.ndarray) -> Dict:
        """
        分析部分依赖

        Args:
            model: 模型
            X_train: 训练数据

        Returns:
            部分依赖分析结果
        """
        # 这里只是示例框架
        return {
            'note': 'Partial dependence analysis implementation would go here',
            'features_analyzed': self.feature_names[:min(5, X_train.shape[1])]
        }

    def _create_lime_global_explanation(self, model, X_train: np.ndarray,
                                        num_samples: int = 10) -> List:
        """
        创建LIME全局解释（多个代表性样本）

        Args:
            model: 模型
            X_train: 训练数据
            num_samples: 样本数量

        Returns:
            LIME解释列表
        """
        explanations = []

        if self.lime_explainer is None:
            self.initialize_lime_explainer(X_train)

        if self.lime_explainer:
            # 选择代表性样本（如聚类中心）
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_samples, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(X_train)

            for cluster_id in range(num_samples):
                # 找到每个聚类的中心样本
                cluster_samples = X_train[cluster_labels == cluster_id]
                if len(cluster_samples) > 0:
                    center_idx = np.argmin(
                        np.linalg.norm(cluster_samples - kmeans.cluster_centers_[cluster_id], axis=1)
                    )
                    sample = cluster_samples[center_idx]

                    # 解释该样本
                    explanation = self.explain_with_lime(
                        model, sample, num_features=8, show_plot=False
                    )
                    if explanation:
                        explanation['cluster_id'] = cluster_id
                        explanation['sample_type'] = 'cluster_center'
                        explanations.append(explanation)

        return explanations

    def _save_global_explanation(self, explanation: Dict, save_path: str):
        """
        保存全局解释

        Args:
            explanation: 解释字典
            save_path: 保存路径
        """
        import json

        # 准备可序列化的解释
        serializable_explanation = {}

        for key, value in explanation.items():
            if key in ['shap_values', 'lime_global']:
                # 跳过复杂对象
                serializable_explanation[key] = f"{type(value).__name__} (not serialized)"
            elif isinstance(value, np.ndarray):
                serializable_explanation[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_explanation[key] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in value.items()
                }
            else:
                serializable_explanation[key] = value

        # 保存为JSON
        json_path = f"{save_path}_global_explanation.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_explanation, f, indent=2, ensure_ascii=False)

        print(f"全局解释保存到: {json_path}")