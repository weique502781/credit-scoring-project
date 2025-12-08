import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


class FeatureEngineer:
    """特征工程类，负责特征创建和选择"""

    def __init__(self):
        self.feature_selector = None

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建新特征"""
        df = data.copy()

        # 示例：根据现有特征创建比率特征（根据实际业务逻辑调整）
        if 'credit_amount' in df.columns and 'duration' in df.columns:
            df['monthly_payment'] = df['credit_amount'] / df['duration']

        # 示例：创建风险比率特征
        if 'existing_credits' in df.columns and 'age' in df.columns:
            df['credit_per_age'] = df['existing_credits'] / df['age']

        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """选择top-k重要特征"""
        self.feature_selector = SelectKBest(f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)

        # 获取选中的特征名称
        selected_mask = self.feature_selector.get_support()
        selected_features = X.columns[selected_mask]
        print(f"选中的特征: {selected_features.tolist()}")

        return pd.DataFrame(X_selected, columns=selected_features)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的选择器转换特征"""
        if self.feature_selector is None:
            raise ValueError("特征选择器尚未拟合，请先调用select_features方法")

        X_selected = self.feature_selector.transform(X)
        selected_mask = self.feature_selector.get_support()
        selected_features = X.columns[selected_mask]

        return pd.DataFrame(X_selected, columns=selected_features)