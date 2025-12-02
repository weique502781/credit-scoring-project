import numpy as np
import pandas as pd
from sklearn.tree import _tree
import json
from typing import List, Dict, Any


class DecisionRulesExtractor:
    """
    从决策树模型中提取可读的决策规则
    """

    def __init__(self, feature_names: List[str], class_names: List[str] = None):
        """
        初始化规则提取器

        Args:
            feature_names: 特征名称列表
            class_names: 类别名称列表，默认为['Bad Credit', 'Good Credit']
        """
        self.feature_names = feature_names

        if class_names is None:
            self.class_names = ['Bad Credit', 'Good Credit']
        else:
            self.class_names = class_names

    def extract_rules_from_tree(self, tree_model, max_depth: int = 5) -> pd.DataFrame:
        """
        从决策树或决策树基模型（如AdaBoost中的树）中提取规则

        Args:
            tree_model: 决策树模型
            max_depth: 最大提取深度

        Returns:
            包含决策规则的DataFrame
        """
        if not hasattr(tree_model, 'tree_'):
            raise ValueError("Model must be a decision tree or have tree_ attribute")

        tree_ = tree_model.tree_
        feature_names = [
            self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        rules = []
        samples = []
        values = []

        def recurse(node, depth, rule):
            """
            递归遍历决策树节点

            Args:
                node: 当前节点索引
                depth: 当前深度
                rule: 当前规则字符串
            """
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # 内部节点
                name = feature_names[node]
                threshold = tree_.threshold[node]

                if depth <= max_depth:
                    # 左子树规则
                    left_rule = f"{rule} AND {name} <= {threshold:.3f}"
                    recurse(tree_.children_left[node], depth + 1, left_rule)

                    # 右子树规则
                    right_rule = f"{rule} AND {name} > {threshold:.3f}"
                    recurse(tree_.children_right[node], depth + 1, right_rule)
            else:
                # 叶节点
                value = tree_.value[node][0]
                total_samples = np.sum(value)
                class_idx = np.argmax(value)
                confidence = value[class_idx] / total_samples
                class_name = self.class_names[class_idx]

                # 清理规则字符串
                rule_clean = rule.strip(' AND ')
                if rule_clean.startswith('AND '):
                    rule_clean = rule_clean[4:]

                rules.append({
                    'rule': rule_clean if rule_clean else 'Root Node',
                    'class': class_name,
                    'confidence': round(confidence, 4),
                    'samples': int(total_samples),
                    'bad_samples': int(value[0]),
                    'good_samples': int(value[1]),
                    'depth': depth
                })

        recurse(0, 1, "")

        # 转换为DataFrame并按置信度排序
        rules_df = pd.DataFrame(rules)
        rules_df = rules_df.sort_values(['confidence', 'samples'], ascending=[False, False])

        # 添加规则ID
        rules_df.insert(0, 'rule_id', range(1, len(rules_df) + 1))

        return rules_df

    def extract_rules_from_adaboost(self, adaboost_model, max_trees: int = 10,
                                    max_depth: int = 3) -> Dict:
        """
        从AdaBoost模型中提取所有弱分类器的规则

        Args:
            adaboost_model: AdaBoost模型
            max_trees: 最大提取树的数量
            max_depth: 每棵树的最大深度

        Returns:
            包含所有树规则的字典
        """
        if not hasattr(adaboost_model, 'estimators_'):
            raise ValueError("Model must be an AdaBoost classifier")

        all_rules = {}

        for i, estimator in enumerate(adaboost_model.estimators_):
            if i >= max_trees:
                break

            try:
                tree_rules = self.extract_rules_from_tree(estimator, max_depth)
                all_rules[f'tree_{i + 1}'] = {
                    'weight': adaboost_model.estimator_weights_[i],
                    'error': adaboost_model.estimator_errors_[i],
                    'rules': tree_rules.to_dict('records') if not tree_rules.empty else []
                }
            except Exception as e:
                print(f"Error extracting rules from tree {i}: {e}")
                continue

        return all_rules

    def simplify_rules(self, rules_df: pd.DataFrame,
                       min_confidence: float = 0.7,
                       min_samples: int = 10) -> pd.DataFrame:
        """
        简化规则，过滤低质量规则

        Args:
            rules_df: 原始规则DataFrame
            min_confidence: 最小置信度阈值
            min_samples: 最小样本数阈值

        Returns:
            简化后的规则DataFrame
        """
        # 过滤规则
        filtered_rules = rules_df[
            (rules_df['confidence'] >= min_confidence) &
            (rules_df['samples'] >= min_samples)
            ].copy()

        # 简化规则表达式
        filtered_rules['rule_simple'] = filtered_rules['rule'].apply(
            self._simplify_rule_expression
        )

        return filtered_rules

    def _simplify_rule_expression(self, rule: str) -> str:
        """
        简化规则表达式，使其更易读

        Args:
            rule: 原始规则字符串

        Returns:
            简化后的规则字符串
        """
        if not rule or rule == 'Root Node':
            return 'All cases'

        # 替换操作符为自然语言
        replacements = {
            ' <= ': ' ≤ ',
            ' > ': ' > ',
            ' AND ': ' AND ',
            'checking_account': 'Checking Account',
            'duration': 'Loan Duration (months)',
            'credit_history': 'Credit History',
            'purpose': 'Loan Purpose',
            'amount': 'Loan Amount',
            'savings': 'Savings Account',
            'employment': 'Employment Status',
            'personal_status': 'Personal Status',
            'debtors': 'Other Debtors',
            'property': 'Property',
            'age': 'Age',
            'other_plans': 'Other Installment Plans',
            'housing': 'Housing',
            'existing_credits': 'Existing Credits',
            'job': 'Job Type',
            'telephone': 'Telephone',
            'foreign_worker': 'Foreign Worker'
        }

        simplified = rule
        for old, new in replacements.items():
            simplified = simplified.replace(old, new)

        return simplified

    def generate_rule_report(self, rules_df: pd.DataFrame,
                             output_path: str = 'reports/decision_rules_report.md') -> str:
        """
        生成规则报告

        Args:
            rules_df: 规则DataFrame
            output_path: 报告保存路径

        Returns:
            Markdown格式的报告字符串
        """
        if rules_df.empty:
            return "No rules found."

        # 统计信息
        total_rules = len(rules_df)
        avg_confidence = rules_df['confidence'].mean()
        avg_samples = rules_df['samples'].mean()
        good_rules = len(rules_df[rules_df['class'] == 'Good Credit'])
        bad_rules = len(rules_df[rules_df['class'] == 'Bad Credit'])

        # 创建Markdown报告
        report = f"""# Decision Rules Report

## Overview
- **Total Rules Extracted:** {total_rules}
- **Average Confidence:** {avg_confidence:.2%}
- **Average Samples per Rule:** {avg_samples:.0f}
- **Rules for Good Credit:** {good_rules}
- **Rules for Bad Credit:** {bad_rules}

## Top Rules for Good Credit
"""

        # 添加好信用规则
        good_credit_rules = rules_df[rules_df['class'] == 'Good Credit'].head(10)
        if not good_credit_rules.empty:
            report += "\n| Rule ID | Rule | Confidence | Samples |\n"
            report += "|---------|------|------------|---------|\n"
            for _, row in good_credit_rules.iterrows():
                report += f"| {row['rule_id']} | {row['rule_simple']} | {row['confidence']:.2%} | {row['samples']} |\n"

        # 添加坏信用规则
        report += "\n## Top Rules for Bad Credit\n"
        bad_credit_rules = rules_df[rules_df['class'] == 'Bad Credit'].head(10)
        if not bad_credit_rules.empty:
            report += "\n| Rule ID | Rule | Confidence | Samples |\n"
            report += "|---------|------|------------|---------|\n"
            for _, row in bad_credit_rules.iterrows():
                report += f"| {row['rule_id']} | {row['rule_simple']} | {row['confidence']:.2%} | {row['samples']} |\n"

        # 添加规则示例
        report += "\n## Rule Examples\n"
        report += "\n### High Confidence Rules (> 90%)\n"
        high_conf_rules = rules_df[rules_df['confidence'] > 0.9].head(5)
        for _, row in high_conf_rules.iterrows():
            report += f"- **IF** {row['rule_simple']} **THEN** {row['class']} "
            report += f"(Confidence: {row['confidence']:.2%}, Samples: {row['samples']})\n"

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Rule report saved to: {output_path}")
        return report

    def export_rules_to_json(self, rules_df: pd.DataFrame,
                             output_path: str = 'reports/decision_rules.json') -> None:
        """
        将规则导出为JSON格式

        Args:
            rules_df: 规则DataFrame
            output_path: 输出文件路径
        """
        # 转换为字典格式
        rules_dict = {
            'metadata': {
                'total_rules': len(rules_df),
                'features_used': list(set([
                    feat.split(' ')[0] for rule in rules_df['rule']
                    for feat in rule.split(' AND ') if feat
                ])),
                'extraction_time': pd.Timestamp.now().isoformat()
            },
            'rules': rules_df.to_dict('records')
        }

        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_dict, f, indent=2, ensure_ascii=False)

        print(f"Rules exported to JSON: {output_path}")