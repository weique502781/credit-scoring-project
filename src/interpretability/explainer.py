import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import shap
from sklearn.tree import plot_tree


class ModelExplainer:
    def __init__(self, feature_names: list, class_names: list = None):
        self.feature_names = feature_names

        if class_names is None:
            self.class_names = ['Bad Credit', 'Good Credit']
        else:
            self.class_names = class_names

    def lime_explain(self, model, X_train: np.ndarray,
                     instance: np.ndarray, num_features: int = 10):
        """使用LIME解释单个预测"""
        # 创建LIME解释器
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=42
        )

        # 生成解释
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=num_features
        )

        # 保存解释
        exp.save_to_file('reports/lime_explanation.html')

        # 在notebook中显示
        # exp.show_in_notebook()

        return exp

    def shap_explain(self, model, X_train: np.ndarray,
                     X_test: np.ndarray = None):
        """使用SHAP进行解释"""
        # 创建SHAP解释器
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.Explainer(model, X_train)
        else:
            explainer = shap.KernelExplainer(model.predict, X_train[:100])

        # 计算SHAP值
        if X_test is None:
            X_explain = X_train[:100]  # 使用部分数据
        else:
            X_explain = X_test

        shap_values = explainer.shap_values(X_explain)

        # 创建多个可视化
        self._create_shap_visualizations(explainer, shap_values, X_explain)

        return explainer, shap_values

    def _create_shap_visualizations(self, explainer, shap_values, X):
        """创建SHAP可视化"""
        # 1. 特征重要性汇总图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig('reports/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 特征重要性条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names,
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('reports/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 单个预测的力图
        if isinstance(shap_values, list):
            shap_values_single = shap_values[1][0]  # 对于二分类
        else:
            shap_values_single = shap_values[0]

        plt.figure(figsize=(12, 6))
        shap.force_plot(explainer.expected_value, shap_values_single,
                        X[0], feature_names=self.feature_names, show=False, matplotlib=True)
        plt.tight_layout()
        plt.savefig('reports/shap_force_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    def extract_decision_rules(self, model, feature_names: list = None):
        """从决策树中提取决策规则"""
        if not hasattr(model, 'tree_'):
            print("Model is not a decision tree!")
            return None

        if feature_names is None:
            feature_names = self.feature_names

        # 提取决策规则
        from sklearn.tree import _tree

        tree_ = model.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        rules = []

        def recurse(node, depth, rule):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # 左子树规则
                left_rule = rule + f"({name} <= {threshold:.2f})"
                recurse(tree_.children_left[node], depth + 1, left_rule)

                # 右子树规则
                right_rule = rule + f"({name} > {threshold:.2f})"
                recurse(tree_.children_right[node], depth + 1, right_rule)
            else:
                # 叶节点
                class_prob = tree_.value[node][0]
                class_idx = np.argmax(class_prob)
                class_name = self.class_names[class_idx]
                confidence = class_prob[class_idx] / np.sum(class_prob)

                rules.append({
                    'rule': rule,
                    'class': class_name,
                    'confidence': confidence,
                    'samples': int(tree_.n_node_samples[node])
                })

        recurse(0, 1, "")

        # 按置信度排序
        rules_sorted = sorted(rules, key=lambda x: x['confidence'], reverse=True)

        # 保存规则
        rules_df = pd.DataFrame(rules_sorted)
        rules_df.to_csv('reports/decision_rules.csv', index=False)

        return rules_df

    def visualize_decision_tree(self, model, feature_names: list = None,
                                max_depth: int = 3):
        """可视化决策树"""
        if not hasattr(model, 'tree_'):
            print("Model is not a decision tree!")
            return

        if feature_names is None:
            feature_names = self.feature_names

        plt.figure(figsize=(20, 10))
        plot_tree(model,
                  feature_names=feature_names,
                  class_names=self.class_names,
                  filled=True,
                  rounded=True,
                  max_depth=max_depth,
                  fontsize=10)

        plt.title('Decision Tree Visualization', fontsize=16)
        plt.tight_layout()
        plt.savefig('reports/decision_tree.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_interpretability_report(self, model, X_train: np.ndarray,
                                       X_test: np.ndarray, instance_idx: int = 0):
        """创建完整的可解释性报告"""
        report = {}

        # LIME解释
        instance = X_test[instance_idx]
        lime_exp = self.lime_explain(model, X_train, instance)
        report['lime'] = lime_exp

        # SHAP解释
        try:
            shap_explainer, shap_values = self.shap_explain(model, X_train, X_test)
            report['shap'] = shap_explainer
        except:
            print("SHAP解释失败，跳过...")

        # 决策规则（如果是树模型）
        if hasattr(model, 'tree_'):
            rules = self.extract_decision_rules(model)
            report['rules'] = rules

            # 可视化决策树
            self.visualize_decision_tree(model)

        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)

            report['feature_importance'] = importance_df

            # 可视化特征重要性
            plt.figure(figsize=(12, 8))
            importance_df.head(15).plot.barh(x='feature', y='importance')
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('reports/feature_importance.png', dpi=300)
            plt.show()

        return report