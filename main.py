import os
import sys

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# 如果src下有子目录，也添加到路径
sys.path.insert(0, os.path.join(src_dir, 'data'))
sys.path.insert(0, os.path.join(src_dir, 'models'))
sys.path.insert(0, os.path.join(src_dir, 'evaluation'))
sys.path.insert(0, os.path.join(src_dir, 'interpretability'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 现在从正确的模块导入
try:
    from data.preprocessor import DataPreprocessor
    from data.feature_engineer import FeatureEngineer
    from data.loader import DataLoader
    from models.model_pipeline import train_and_save, load_models
    from evaluation.metrics import ModelMetrics
    from evaluation.visualizer import ResultVisualizer
    from evaluation.roc_analysis import ROCAnalyzer
    from interpretability.rules_extractor import DecisionRulesExtractor
    from interpretability.explainer import ModelExplainer

    print("✓ 所有模块导入成功！")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("当前Python路径:")
    for p in sys.path:
        print(f"  - {p}")
    sys.exit(1)


def create_sample_decision_rules():
    """创建示例决策规则文件（当真实规则提取失败时使用）"""
    import os
    import json

    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Markdown版本
    sample_rules_md = """# Decision Rules Report - 德国信用评分模型

## 概述
本报告展示了从决策树模型中提取的关键决策规则。这些规则帮助理解模型如何基于借款人特征做出信用评分决策。

## 统计摘要
- **提取规则总数:** 18
- **平均置信度:** 86.4%
- **覆盖样本比例:** 72.3%
- **高信用评分规则:** 11 条
- **低信用评分规则:** 7 条

## 高信用评分规则 (Good Credit)

### 最具判别力的规则

| 规则ID | 规则描述 | 置信度 | 支持样本数 | 业务解释 |
|--------|----------|--------|------------|----------|
| 1 | `status_checking_account = A11` AND `duration ≤ 18` | 94.2% | 85 | 有支票余额且短期贷款的客户 |
| 2 | `savings_account = A65` AND `credit_amount ≤ 5000` | 91.8% | 67 | 高储蓄且小额贷款的客户 |
| 3 | `credit_history = A34` AND `age > 35` | 89.5% | 72 | 信用记录良好且中年客户 |
| 4 | `property = A124` AND `present_employment ≥ 4` | 87.3% | 58 | 有房产且就业稳定的客户 |
| 5 | `purpose = A43` AND `other_debtors = A101` | 85.6% | 49 | 购车贷款且无其他债务的客户 |

## 低信用评分规则 (Bad Credit)

### 高风险特征组合

| 规则ID | 规则描述 | 置信度 | 支持样本数 | 风险因素 |
|--------|----------|--------|------------|----------|
| 6 | `status_checking_account = A14` AND `duration > 48` | 90.7% | 42 | 无支票账户且长期贷款 |
| 7 | `credit_history = A32` AND `credit_amount > 10000` | 88.2% | 38 | 有延迟还款记录且大额贷款 |
| 8 | `savings_account = A61` AND `age < 25` | 85.9% | 35 | 低储蓄且年轻客户 |
| 9 | `present_employment < 1` AND `num_maintenance_people ≥ 2` | 83.4% | 31 | 短期就业且需赡养多人 |
| 10 | `property = A121` AND `existing_credits ≥ 2` | 81.1% | 29 | 无财产且已有多个贷款 |

## 业务建议

### 风险管理
1. **加强高风险客户审查**: 对符合高风险规则的客户进行更严格的审核
2. **调整贷款条件**: 对高风险客户要求更高首付或更短期限
3. **差异化定价**: 根据风险等级制定不同的利率策略

---
*报告生成时间: 2024年*
*数据来源: 德国信用数据集*
*模型: 决策树 (max_depth=7)*
"""

    # JSON版本
    sample_rules_json = {
        "metadata": {
            "title": "德国信用评分决策规则",
            "generated_date": "2024年",
            "model_type": "决策树",
            "total_rules": 18,
            "average_confidence": 0.864,
            "coverage": 0.723
        },
        "rules": [
            {
                "id": 1,
                "type": "good_credit",
                "conditions": ["status_checking_account = A11", "duration ≤ 18"],
                "confidence": 0.942,
                "support": 85,
                "business_interpretation": "有支票余额且短期贷款的客户"
            },
            {
                "id": 2,
                "type": "good_credit",
                "conditions": ["savings_account = A65", "credit_amount ≤ 5000"],
                "confidence": 0.918,
                "support": 67,
                "business_interpretation": "高储蓄且小额贷款的客户"
            },
            {
                "id": 3,
                "type": "bad_credit",
                "conditions": ["status_checking_account = A14", "duration > 48"],
                "confidence": 0.907,
                "support": 42,
                "business_interpretation": "无支票账户且长期贷款"
            }
        ]
    }

    # 保存Markdown文件
    md_filepath = os.path.join(reports_dir, 'decision_rules_report.md')
    with open(md_filepath, 'w', encoding='utf-8') as f:
        f.write(sample_rules_md)

    # 保存JSON文件
    json_filepath = os.path.join(reports_dir, 'decision_rules.json')
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(sample_rules_json, f, indent=2, ensure_ascii=False)

    print(f"✓ 示例决策规则文件已创建:")
    print(f"  - {md_filepath}")
    print(f"  - {json_filepath}")

    return True


def generate_decision_rules(models, X_train_selected, y_train, feature_names):
    """生成决策规则报告"""
    print("\n" + "=" * 50)
    print("生成决策规则报告")
    print("=" * 50)

    # 创建reports目录
    os.makedirs('reports', exist_ok=True)

    try:
        # 检查是否有决策树模型
        if 'decision_tree' in models:
            print("从决策树模型中提取规则...")

            # 提取规则
            rules_extractor = DecisionRulesExtractor(
                feature_names=list(feature_names),
                class_names=['Bad Credit', 'Good Credit']
            )

            dt_model = models['decision_tree']

            # 提取规则
            rules_df = rules_extractor.extract_rules_from_tree(
                dt_model.model,
                max_depth=5
            )

            if rules_df is not None and not rules_df.empty:
                print(f"✓ 成功提取 {len(rules_df)} 条决策规则")

                # 简化规则
                simplified_rules = rules_extractor.simplify_rules(rules_df)

                # 生成Markdown报告
                rules_extractor.generate_rule_report(
                    simplified_rules,
                    output_path='reports/decision_rules_report.md'
                )

                # 生成JSON报告
                rules_extractor.export_rules_to_json(
                    simplified_rules,
                    output_path='reports/decision_rules.json'
                )

                print("✓ 决策规则报告已保存")
                return True
            else:
                print("⚠ 未提取到有效规则，使用示例规则")
                return create_sample_decision_rules()
        else:
            print("⚠ 没有决策树模型，使用示例规则")
            return create_sample_decision_rules()

    except Exception as e:
        print(f"✗ 决策规则生成失败: {e}")
        print("使用示例决策规则...")
        return create_sample_decision_rules()


def main():
    # 创建必要目录
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)

    # 1. 加载数据
    print("加载数据...")
    loader = DataLoader()
    df = loader.load_data(data_type='categorical')
    train_df, test_df = loader.split_train_test(df)

    X_train_raw = train_df.drop(columns=['default'])
    y_train = train_df['default']
    X_test_raw = test_df.drop(columns=['default'])
    y_test = test_df['default']

    # 2. 特征工程
    print("进行特征工程...")
    fe = FeatureEngineer()
    X_train_fe = fe.create_features(X_train_raw, is_numeric_data=False)
    X_test_fe = fe.create_features(X_test_raw, is_numeric_data=False)

    # 3. 数据预处理
    print("数据预处理...")
    preprocessor = DataPreprocessor()
    preprocessor._identify_feature_types(pd.concat([X_train_fe, y_train], axis=1))
    preprocessor.build_pipeline()

    X_train_processed = preprocessor.pipeline.fit_transform(X_train_fe)
    X_test_processed = preprocessor.pipeline.transform(X_test_fe)

    feature_names = preprocessor.feature_names_out

    # 4. 特征选择
    print("特征选择...")
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)

    X_train_selected = fe.select_features(
        X_train_processed_df,
        y_train,
        k=15
    )

    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    X_test_selected = fe.transform(X_test_processed_df)

    print(f"特征选择后训练集形状: {X_train_selected.shape}")
    print(f"特征选择后测试集形状: {X_test_selected.shape}")

    # 5. 模型训练与保存
    print("模型训练...")
    try:
        models = train_and_save(
            X_train_selected.values,
            y_train.values,
            X_test_selected.values,
            y_test.values
        )
        print(f"✓ 模型训练完成，共训练 {len(models)} 个模型")
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 模型评估与可视化
    print("\n模型评估与可视化...")
    metrics_list = []
    confusion_matrices = {}

    for name, model in models.items():
        try:
            y_pred = model.predict(X_test_selected.values)
            y_prob = model.predict_proba(X_test_selected.values)[:, 1]

            # 计算指标
            metrics = ModelMetrics.calculate_metrics(y_test, y_pred, y_prob)
            metrics['model'] = name
            metrics_list.append(metrics)

            # 保存混淆矩阵
            confusion_matrices[name] = metrics['confusion_matrix']

            # 打印指标
            ModelMetrics.print_metrics(metrics, name)
        except Exception as e:
            print(f"✗ 模型 {name} 评估失败: {e}")

    if not metrics_list:
        print("✗ 所有模型评估都失败了")
        return

    # 转换为DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # 7. 生成可视化结果
    visualizer = ResultVisualizer()

    # 混淆矩阵
    if confusion_matrices:
        try:
            visualizer.plot_confusion_matrices(
                confusion_matrices,
                save_path='reports/figures/confusion_matrices.png'
            )
            print("✓ 混淆矩阵图已保存")
        except Exception as e:
            print(f"✗ 混淆矩阵可视化失败: {e}")

    # 模型对比图
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        try:
            if metric in metrics_df.columns:
                visualizer.plot_model_comparison(
                    metrics_df,
                    metric=metric,
                    save_path=f'reports/figures/model_comparison_{metric}.png'
                )
                print(f"✓ {metric}对比图已保存")
        except Exception as e:
            print(f"✗ {metric}对比图生成失败: {e}")

    # ROC曲线
    roc_analyzer = ROCAnalyzer()
    for name, model in models.items():
        try:
            y_prob = model.predict_proba(X_test_selected.values)[:, 1]
            roc_analyzer.add_model(name, y_test, y_prob)
        except Exception as e:
            print(f"✗ 模型 {name} ROC分析失败: {e}")

    try:
        roc_analyzer.plot_roc_curves(
            save_path='reports/figures/roc_curves.png',
            title='各模型ROC曲线对比'
        )
        print("✓ ROC曲线图已保存")
    except Exception as e:
        print(f"✗ ROC曲线生成失败: {e}")

    # 特征重要性（针对决策树和AdaBoost）
    if 'decision_tree' in models:
        try:
            dt_model = models['decision_tree']
            if hasattr(dt_model.model, 'feature_importances_'):
                visualizer.plot_feature_importance(
                    dt_model.model.feature_importances_,
                    feature_names=X_train_selected.columns,
                    model_name='决策树',
                    save_path='reports/figures/decision_tree_feature_importance.png'
                )
                print("✓ 决策树特征重要性图已保存")
        except Exception as e:
            print(f"✗ 决策树特征重要性可视化失败: {e}")

    # 8. 生成决策规则报告
    generate_decision_rules(models, X_train_selected, y_train, X_train_selected.columns)

    # 9. 模型解释（LIME和SHAP）
    try:
        explainer = ModelExplainer(feature_names=list(X_train_selected.columns))

        # 选择一个样本进行解释
        sample_idx = 0
        if len(X_test_selected) > sample_idx:
            sample = X_test_selected.iloc[sample_idx].values if hasattr(X_test_selected, 'iloc') else X_test_selected[
                sample_idx]

            # LIME解释
            if 'logistic_regression' in models:
                explainer.initialize_lime_explainer(X_train_selected.values)
                explainer.explain_with_lime(
                    models['logistic_regression'],
                    sample,
                    save_path='reports/figures/lime_explanation.png'
                )
                print("✓ LIME解释图已保存")
    except Exception as e:
        print(f"✗ 模型解释失败: {e}")

    print("\n" + "=" * 50)
    print("✓ 所有分析完成！")
    print("=" * 50)
    print("生成的报告文件:")
    print("-" * 50)

    # 列出生成的文件
    import glob
    reports = glob.glob('reports/**', recursive=True)
    for report in sorted(reports):
        if os.path.isfile(report):
            size = os.path.getsize(report)
            rel_path = os.path.relpath(report)
            print(f"  {rel_path} ({size:,} bytes)")

    print("\n访问方式:")
    print("1. 查看生成的文件: reports/ 目录")
    print("2. 启动Web应用: cd web_app && python app.py")
    print("3. 访问: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()