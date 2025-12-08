import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pickle
import os


class DataPreprocessor:
    """数据预处理管道，处理缺失值、编码和标准化"""

    def __init__(self):
        self.pipeline = None
        self.categorical_features = None
        self.numerical_features = None

    def _identify_feature_types(self, data: pd.DataFrame, target_col: str) -> None:
        """识别分类特征和数值特征"""
        # 排除目标列
        features = data.drop(columns=[target_col])
        # 分类特征（object或category类型）
        self.categorical_features = list(features.select_dtypes(include=['object', 'category']).columns)
        # 数值特征（int或float类型）
        self.numerical_features = list(features.select_dtypes(include=['int64', 'float64']).columns)
        print(f"分类特征: {self.categorical_features}")
        print(f"数值特征: {self.numerical_features}")

    def build_pipeline(self) -> None:
        """构建预处理管道"""
        # 数值特征处理：填充缺失值+标准化
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 分类特征处理：填充缺失值+独热编码
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 组合所有特征处理
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def fit(self, data: pd.DataFrame, target_col: str = 'default') -> None:
        """拟合预处理管道"""
        self._identify_feature_types(data, target_col)
        self.build_pipeline()
        X = data.drop(columns=[target_col])
        self.pipeline.fit(X)

    def transform(self, data: pd.DataFrame, target_col: str = 'default') -> tuple:
        """转换数据并返回特征和目标变量"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        X_processed = self.pipeline.transform(X)
        return X_processed, y

    def save_pipeline(self, save_path: str) -> None:
        """保存预处理管道"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"预处理管道已保存至: {save_path}")

    @staticmethod
    def load_pipeline(load_path: str) -> 'DataPreprocessor':
        """加载预处理管道"""
        with open(load_path, 'rb') as f:
            return pickle.load(f)