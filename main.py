# main.py

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from src.data.loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.models.model_pipeline import train_and_save


def main():
    print("\n" + "=" * 60)
    print("信用评分模型 - 最终模型训练")
    print("=" * 60)

    print(f"\n 使用优化配置:")
    print(f"   - 所有基础模型均为优化版本")
    print(f"   - 使用改进版AdaBoost")
    print(f"   - 使用优化参数配置")

    # 1. 加载数据
    print("\n[1/5] 加载原始数据...")
    loader = DataLoader()
    df = loader.load_data("categorical")

    # 显示数据信息
    print(f"数据集形状: {df.shape}")
    print(f"违约分布:")
    print(df['default'].value_counts(normalize=True))

    # 2. 划分数据集
    print("\n[2/5] 划分训练/测试集...")
    train_df, test_df = loader.split_train_test(df)
    print(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")

    # 3. 特征工程
    print("\n[3/5] 特征工程...")
    fe = FeatureEngineer()
    train_df = fe.create_features(train_df)
    test_df = fe.create_features(test_df)
    print(f"特征工程后特征数: {train_df.shape[1] - 1}")

    # 4. 数据预处理
    print("\n[4/5] 数据预处理...")
    pre = DataPreprocessor()
    X_train, y_train = pre.fit_transform(train_df)
    X_test, y_test = pre.transform(test_df)

    feature_names = pre.feature_names_out
    print(f"预处理后训练集形状: {X_train.shape}")
    print(f"预处理后测试集形状: {X_test.shape}")

    # 5. 使用新的训练流水线
    print("\n[5/5] 训练并保存所有模型...")

    # 使用新的 train_and_save 函数
    models = train_and_save(
        X_train, y_train,
        X_test, y_test,
        save_dir="models/saved_models",
        random_state=42
    )

    # 加载模型进行特征重要性分析
    print("\n" + "=" * 40)
    print("特征重要性分析")
    print("=" * 40)

    # 查找AdaBoost模型进行特征重要性分析
    ada_model = None
    for name, model in models.items():
        if 'adaboost' in name.lower() and hasattr(model, 'get_feature_importances'):
            ada_model = model
            print(f"使用模型进行特征重要性分析: {name}")
            break

    if ada_model and hasattr(ada_model, 'get_feature_importances'):
        print("\nAdaBoost 特征重要性 Top10:")
        print("-" * 40)

        try:
            importances = ada_model.get_feature_importances(feature_names)
            top_features = dict(sorted(importances.items(),
                                       key=lambda x: x[1],
                                       reverse=True)[:10])

            for feature, importance in top_features.items():
                print(f"{feature:30s}: {importance:.4f}")

            # 保存特征重要性到CSV
            importance_df = pd.DataFrame(
                list(importances.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)

            save_dir = Path("models/saved_models")
            importance_path = save_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
            print(f"\n 特征重要性已保存到: {importance_path}")

        except Exception as e:
            print(f"获取特征重要性失败: {e}")

    # 6. 保存额外的文件（预处理器和特征名称）
    print("\n" + "=" * 40)
    print("保存额外文件")
    print("=" * 40)

    save_dir = Path("models/saved_models")

    # 保存预处理器
    preprocessor_path = save_dir / "preprocessor.pkl"
    joblib.dump(pre, preprocessor_path)
    print(f"✓ preprocessor.pkl")

    # 保存特征名称
    feature_names_path = save_dir / "feature_names.npy"
    np.save(feature_names_path, feature_names)
    print(f"✓ feature_names.npy")

    # 保存完整的训练配置
    from src.models.ensemble import EnsembleTrainer
    trainer = EnsembleTrainer(random_state=42)
    trainer.models = models  # 使用训练好的模型

    # 在测试集上评估
    test_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        test_results[name] = accuracy

    train_config = {
        'model_config': 'optimized_models_only',
        'random_state': 42,
        'feature_count': len(feature_names),
        'models_trained': list(models.keys()),
        'model_details': {
            'logistic_regression': '优化参数(C=0.5, solver=liblinear)',
            'decision_tree': '优化参数(max_depth=7, max_features=sqrt)',
            'svm': '优化参数(C=2.0, gamma=0.1, cache_size=500)',
            'naive_bayes': '高斯朴素贝叶斯',
            'custom_adaboost': '改进版AdaBoost(early_stopping, bootstrap)',
            'sklearn_adaboost': 'sklearn官方AdaBoost'
        },
        'test_accuracy': test_results,
        'avg_accuracy': np.mean(list(test_results.values())),
        'best_model': max(test_results.items(), key=lambda x: x[1])[0]
    }

    config_path = save_dir / "train_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)
    print(f"✓ train_config.json")

    # 生成最终报告
    print("\n" + "=" * 60)
    print(" 训练完成!")
    print("=" * 60)

    # 列出所有生成的文件
    print(f"保存的模型文件 ({save_dir}):")
    all_files = []

    # 基础模型文件
    for file in save_dir.glob("*.pkl"):
        if file.name != "preprocessor.pkl":
            all_files.append(file.name)
            print(f"  • {file.name}")

    # 集成模型文件
    ensemble_dir = save_dir / "ensemble"
    if ensemble_dir.exists():
        print(f"  • ensemble/")
        for file in ensemble_dir.glob("*.pkl"):
            all_files.append(f"ensemble/{file.name}")
            print(f"    └── {file.name}")

    # 其他文件
    other_files = ["preprocessor.pkl", "feature_names.npy", "train_config.json", "feature_importance.csv"]
    for file in other_files:
        if (save_dir / file).exists():
            all_files.append(file)
            print(f"  • {file}")

    print(f"\n 模型性能排名:")
    results_df = pd.DataFrame({
        '模型': list(test_results.keys()),
        '准确率': list(test_results.values())
    }).sort_values('准确率', ascending=False)

    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]
    print(f"\n 最佳模型: {best_model['模型']} (准确率: {best_model['准确率']:.4f})")

    avg_accuracy = results_df['准确率'].mean()
    std_accuracy = results_df['准确率'].std()
    print(f"\n 整体统计:")
    print(f"   平均准确率: {avg_accuracy:.4f}")
    print(f"   准确率标准差: {std_accuracy:.4f}")
    print(f"   模型数量: {len(results_df)}")

if __name__ == "__main__":
    main()