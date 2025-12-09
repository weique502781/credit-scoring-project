from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pickle
import os


class DataPreprocessor:
    """数据预处理管道，处理缺失值、编码和标准化（适配德国信用数据集）"""

    def __init__(self):
        self.pipeline = None
        self.categorical_features = None
        self.numerical_features = None
        self.feature_names_out = None  # 存储处理后的特征名

        # 新增：动态定位项目根目录，确保保存路径正确
        current_script_path = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
        self.default_save_dir = os.path.join(self.project_root, 'models', 'saved_models')
        os.makedirs(self.default_save_dir, exist_ok=True)

    def _identify_feature_types(self, data: pd.DataFrame, target_col: str = 'default') -> None:
        """识别分类特征和数值特征（适配德国信用数据集）"""
        # 排除目标列
        features = data.drop(columns=[target_col])
        # 分类特征（object或category类型）
        self.categorical_features = list(features.select_dtypes(include=['object', 'category']).columns)
        # 数值特征（int或float类型）
        self.numerical_features = list(features.select_dtypes(include=['int64', 'float64']).columns)

        # 特殊处理：如果是numeric类型数据集（feature_1~feature_24），全部视为数值特征
        if all(col.startswith('feature_') for col in features.columns):
            self.numerical_features = list(features.columns)
            self.categorical_features = []

        print(f"分类特征: {self.categorical_features}")
        print(f"数值特征: {self.numerical_features}")

    def build_pipeline(self) -> None:
        """构建预处理管道"""
        # 数值特征处理：填充缺失值（中位数）+ 标准化
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 分类特征处理：填充缺失值（众数）+ 独热编码（忽略未知类别）
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # 组合所有特征处理
        transformers = []
        if self.numerical_features:
            transformers.append(('num', numerical_transformer, self.numerical_features))
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))

        self.pipeline = ColumnTransformer(transformers=transformers)

    def fit(self, data: pd.DataFrame, target_col: str = 'default') -> None:
        """拟合预处理管道"""
        self._identify_feature_types(data, target_col)
        self.build_pipeline()
        X = data.drop(columns=[target_col])
        self.pipeline.fit(X)

        # 保存处理后的特征名（用于后续可解释性分析）
        self._get_feature_names_out()

    def transform(self, data: pd.DataFrame, target_col: str = 'default') -> tuple[np.ndarray, np.ndarray]:
        """转换数据并返回特征（numpy数组）和目标变量（numpy数组）"""
        X = data.drop(columns=[target_col])
        y = data[target_col].values
        X_processed = self.pipeline.transform(X)
        return X_processed, y

    def fit_transform(self, data: pd.DataFrame, target_col: str = 'default') -> tuple[np.ndarray, np.ndarray]:
        """拟合并转换数据"""
        self.fit(data, target_col)
        return self.transform(data, target_col)

    def _get_feature_names_out(self) -> None:
        """获取处理后的特征名（适配独热编码后的分类特征）"""
        feature_names = []
        # 数值特征名
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        # 分类特征名（独热编码后）
        if self.categorical_features:
            ohe = self.pipeline.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
        self.feature_names_out = feature_names

    def save_pipeline(self, save_path: str = None) -> None:
        """保存预处理管道到项目根目录的 models/saved_models/"""
        if save_path is None:
            save_path = os.path.join(self.default_save_dir, 'preprocessor.pkl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"预处理管道已保存至（绝对路径）: {os.path.abspath(save_path)}")

    @staticmethod
    def load_pipeline(load_path: str = None) -> 'DataPreprocessor':
        """从项目根目录的 models/saved_models/ 加载预处理管道"""
        if load_path is None:
            current_script_path = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
            load_path = os.path.join(project_root, 'models', 'saved_models', 'preprocessor.pkl')

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"预处理管道文件不存在：{os.path.abspath(load_path)}")

        with open(load_path, 'rb') as f:
            return pickle.load(f)


# 测试代码
if __name__ == "__main__":
    from loader import DataLoader

    # 加载数据
    loader = DataLoader()
    df = loader.load_data(data_type='categorical')
    train_df, test_df = loader.split_train_test(df)

    # 初始化并拟合预处理管道
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)

    # 转换测试集
    X_test, y_test = preprocessor.transform(test_df)

    print(f"处理后训练集特征形状：{X_train.shape}")
    print(f"处理后特征名数量：{len(preprocessor.feature_names_out)}")
    print(f"前5个特征名：{preprocessor.feature_names_out[:5]}")

    # 保存管道
    preprocessor.save_pipeline()