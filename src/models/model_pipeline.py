# src/models/model_pipeline.py

import os
import joblib
import numpy as np
from typing import Dict, Optional
from sklearn.model_selection import train_test_split

from .ensemble import EnsembleTrainer


def train_and_save(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: Optional[np.ndarray] = None,
                   y_test: Optional[np.ndarray] = None,
                   save_dir: str = "models/saved_models",
                   random_state: int = 42) -> Dict[str, object]:
    """
    训练流水线代码 - 一行代码完成训练 + 保存

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征（可选）
        y_test: 测试标签（可选）
        save_dir: 保存目录
        random_state: 随机种子

    Returns:
        训练好的模型字典
    """
    print("=" * 50)
    print("模型训练流水线")
    print("=" * 50)

    # 1. 如果没提供测试集，从训练集划分
    if X_test is None or y_test is None:
        print("[Pipeline] 从训练集划分20%作为测试集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=random_state,
            stratify=y_train
        )
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 2. 训练所有模型
    print("\n[Pipeline] 开始训练6个模型...")
    trainer = EnsembleTrainer(random_state=random_state)
    trainer.train_all(X_train, y_train)
    models = trainer.get_models()

    # 3. 评估模型性能
    print("\n[Pipeline] 测试集性能评估:")
    results = trainer.quick_evaluate(X_test, y_test)

    # 计算平均准确率
    avg_accuracy = np.mean(list(results.values()))
    print(f"\n 平均准确率: {avg_accuracy:.4f}")

    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1])
    print(f" 最佳模型: {best_model[0]} (准确率: {best_model[1]:.4f})")

    # 4. 保存模型
    print("\n[Pipeline] 保存模型文件...")
    os.makedirs(save_dir, exist_ok=True)

    # 保存基础模型
    base_models_dir = save_dir
    ensemble_dir = os.path.join(save_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)

    model_files = []

    # 保存4个基础模型
    for name in ['logistic_regression', 'decision_tree', 'svm_rbf', 'naive_bayes']:
        if name in models:
            file_path = os.path.join(base_models_dir, f"{name.replace('_rbf', '')}.pkl")
            joblib.dump(models[name], file_path)
            model_files.append(file_path)
            print(f"✓ {os.path.basename(file_path)}")

    # 保存集成模型到 ensemble 目录
    for name in ['adaboost_custom', 'sklearn_adaboost']:
        if name in models:
            # 根据项目要求重命名
            new_name = 'custom_adaboost' if name == 'adaboost_custom' else name
            file_path = os.path.join(ensemble_dir, f"{new_name}.pkl")
            joblib.dump(models[name], file_path)
            model_files.append(file_path)
            print(f"✓ ensemble/{os.path.basename(file_path)}")

    print(f"\n 训练流水线完成！")
    print(f"   共训练了 {len(models)} 个模型")
    print(f"   保存了 {len(model_files)} 个模型文件")
    print(f"   平均准确率: {avg_accuracy:.4f}")

    return models


def load_models(model_dir: str = "models/saved_models") -> Dict[str, object]:
    """
    加载训练好的模型

    Args:
        model_dir: 模型目录

    Returns:
        加载的模型字典
    """
    import glob

    models = {}

    # 加载基础模型
    base_model_files = glob.glob(f"{model_dir}/*.pkl")
    for file_path in base_model_files:
        model_name = os.path.basename(file_path).replace('.pkl', '')
        try:
            model = joblib.load(file_path)
            models[model_name] = model
            print(f"✓ 加载模型: {model_name}")
        except Exception as e:
            print(f" 加载模型 {model_name} 失败: {e}")

    # 加载集成模型
    ensemble_model_files = glob.glob(f"{model_dir}/ensemble/*.pkl")
    for file_path in ensemble_model_files:
        model_name = os.path.basename(file_path).replace('.pkl', '')
        try:
            model = joblib.load(file_path)
            models[model_name] = model
            print(f"✓ 加载集成模型: {model_name}")
        except Exception as e:
            print(f" 加载集成模型 {model_name} 失败: {e}")

    print(f"\n 共加载 {len(models)} 个模型")
    return models


# 保持向后兼容的简单函数
def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """
    训练所有模型（简化版，用于同学B调用）

    Args:
        X_train: 训练特征
        y_train: 训练标签

    Returns:
        训练好的模型字典
    """
    return train_and_save(X_train, y_train)