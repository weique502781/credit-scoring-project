import os
import sys

# 添加 src 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

sys.path.insert(0, os.path.join(src_dir, 'data'))
sys.path.insert(0, os.path.join(src_dir, 'models'))
sys.path.insert(0, os.path.join(src_dir, 'evaluation'))
sys.path.insert(0, os.path.join(src_dir, 'interpretability'))

import numpy as np
import pandas as pd

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
    sys.exit(1)

# 决策规则生成
def generate_decision_rules(models, X_train_selected, y_train, feature_names):
    print("\n" + "=" * 50)
    print("生成决策规则报告")
    print("=" * 50)

    os.makedirs('reports', exist_ok=True)

    try:
        if 'decision_tree' not in models:
            print("⚠ 没有决策树模型，跳过规则生成")
            return

        extractor = DecisionRulesExtractor(
            feature_names=list(feature_names),
            class_names=['Bad Credit', 'Good Credit']
        )

        rules_df = extractor.extract_rules_from_tree(
            models['decision_tree'].model,
            max_depth=5
        )

        if rules_df is None or rules_df.empty:
            print("⚠ 未提取到有效规则")
            return

        simplified = extractor.simplify_rules(rules_df)

        extractor.generate_rule_report(
            simplified,
            output_path='reports/decision_rules_report.md'
        )
        extractor.export_rules_to_json(
            simplified,
            output_path='reports/decision_rules.json'
        )

        print("✓ 决策规则报告已生成")

    except Exception as e:
        print(f"✗ 决策规则生成失败: {e}")



# 主流程

def main():
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    
    # 1. 数据加载
    print("加载数据...")
    loader = DataLoader()
    df = loader.load_data(data_type='categorical')
    train_df, test_df = loader.split_train_test(df)

    X_train_raw = train_df.drop(columns=['default'])
    y_train = train_df['default']
    X_test_raw = test_df.drop(columns=['default'])
    y_test = test_df['default']

    # 2. 特征工程
    print("特征工程...")
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
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_train_selected = fe.select_features(X_train_df, y_train, k=15)

    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    X_test_selected = fe.transform(X_test_df)

    
    # 5. 模型训练（含：权重 + 最优阈值 + final_ensemble）
    print("模型训练...")
    trained_models = train_and_save(
        X_train_selected.values,
        y_train.values,
        X_test_selected.values,
        y_test.values,
        weight_strategy="auc_f1"
    )

    # 6. 加载最终集成模型（系统输出）
    model_bundle = load_models("models/saved_models")
    final_ensemble = model_bundle.get("final_ensemble")

    if final_ensemble is None:
        print("✗ final_ensemble 未找到，程序终止")
        return

    print(f"✓ 已加载 final_ensemble（阈值={final_ensemble.threshold:.3f}）")

    # 7. 单模型评估
    print("\n模型评估...")
    metrics_list = []
    confusion_matrices = {}

    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test_selected.values)
            y_prob = model.predict_proba(X_test_selected.values)[:, 1]

            metrics = ModelMetrics.calculate_metrics(y_test, y_pred, y_prob)
            metrics['model'] = name

            metrics_list.append(metrics)
            confusion_matrices[name] = metrics['confusion_matrix']

            ModelMetrics.print_metrics(metrics, name)
        except Exception as e:
            print(f"✗ 模型 {name} 评估失败: {e}")

    
    # 8. 最终集成模型评估（系统最终输出）
    print("\n最终集成模型评估（Final Ensemble）")

    y_pred_ens = final_ensemble.predict(X_test_selected.values)
    y_prob_ens = final_ensemble.predict_proba(X_test_selected.values)[:, 1]

    ens_metrics = ModelMetrics.calculate_metrics(y_test, y_pred_ens, y_prob_ens)
    ens_metrics['model'] = 'final_ensemble'

    metrics_list.append(ens_metrics)
    confusion_matrices['final_ensemble'] = ens_metrics['confusion_matrix']

    ModelMetrics.print_metrics(ens_metrics, 'final_ensemble')

    metrics_df = pd.DataFrame(metrics_list)

    
    # 9. 可视化
    visualizer = ResultVisualizer()

    visualizer.plot_confusion_matrices(
        confusion_matrices,
        save_path='reports/figures/confusion_matrices.png'
    )

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        visualizer.plot_model_comparison(
            metrics_df,
            metric=metric,
            save_path=f'reports/figures/model_comparison_{metric}.png'
        )

    
    # 10. ROC 曲线
    roc = ROCAnalyzer()

    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            roc.add_model(
                name,
                y_test,
                model.predict_proba(X_test_selected.values)[:, 1]
            )

    roc.add_model("final_ensemble", y_test, y_prob_ens)

    roc.plot_roc_curves(
        save_path='reports/figures/roc_curves.png',
        title='ROC 曲线对比（含最终集成模型）'
    )
    
    # 11. 决策规则
    generate_decision_rules(
        trained_models,
        X_train_selected,
        y_train,
        X_train_selected.columns
    )

    print("\n" + "=" * 50)
    print("全流程完成（final_ensemble 为系统最终输出）")
    print("=" * 50)

if __name__ == "__main__":
    main()
