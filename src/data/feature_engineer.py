from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Optional


class FeatureEngineer:
    """特征工程类（适配德国信用数据集，基于业务逻辑创建特征）"""

    def __init__(self):
        self.feature_selector = None
        self.selected_features = None  # 存储选中的特征名

    def create_features(self, data: pd.DataFrame, is_numeric_data: bool = False) -> pd.DataFrame:
        """
        基于德国信用数据集业务逻辑创建新特征
        :param data: 原始特征DataFrame（未去除目标列）
        :param is_numeric_data: 是否为全数值数据集（data_type='numeric'）
        :return: 新增特征后的DataFrame
        """
        df = data.copy()

        # 一、针对原始categorical数据集的业务特征（基于german.doc的特征定义）
        if not is_numeric_data:
            # 1. 月供能力：贷款金额 / 贷款期限（月）
            if 'credit_amount' in df.columns and 'duration' in df.columns:
                df['monthly_payment'] = df['credit_amount'] / df['duration']
                df['monthly_payment'] = df['monthly_payment'].round(2)  # 保留2位小数

            # 2. 信用密度：现有贷款数 / 年龄（反映年龄与负债的关系）
            if 'existing_credits' in df.columns and 'age' in df.columns:
                df['credit_per_age'] = df['existing_credits'] / df['age']
                df['credit_per_age'] = df['credit_per_age'].round(4)

            # 3. 收入压力：月供占收入比例 * 需赡养人数（反映家庭负担）
            if 'installment_rate' in df.columns and 'num_maintenance_people' in df.columns:
                df['income_pressure'] = df['installment_rate'] * df['num_maintenance_people']

            # 4. 居住稳定性：居住年限 / 年龄（反映扎根程度）
            if 'present_residence' in df.columns and 'age' in df.columns:
                df['residence_stability'] = df['present_residence'] / df['age']
                df['residence_stability'] = df['residence_stability'].round(4)

            # 5. 信用历史风险标记：有逾期记录则为1（基于credit_history特征）
            if 'credit_history' in df.columns:
                risky_credit_history = ['A33', 'A34']  # 延迟还款/严重账户
                df['has_risky_credit'] = df['credit_history'].isin(risky_credit_history).astype(int)

        # 二、通用特征（适用于所有数据类型）
        # 1. 贷款金额分位数标记（按四分位数）
        if 'credit_amount' in df.columns:
            df['credit_amount_quantile'] = pd.qcut(
                df['credit_amount'], q=4, labels=[0, 1, 2, 3]
            ).astype(int)

        # 2. 年龄分组：青年(≤30)、中年(31-50)、老年(>50)
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]
            ).astype(int)

        print(f"特征工程后数据形状：{df.shape}")
        print(f"新增特征：{[col for col in df.columns if col not in data.columns]}")
        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: Optional[int] = None) -> pd.DataFrame:
        """选择top-k重要特征（适配德国信用数据集）"""
        # 若未指定k，自动选择显著特征（p值<0.05）
        if k is None:
            self.feature_selector = SelectKBest(f_classif, k='all')
            X_scored = self.feature_selector.fit_transform(X, y)  # 这里已经是全部特征（k='all'）
            p_values = self.feature_selector.pvalues_
            self.selected_features = X.columns[p_values < 0.05].tolist()
            k = len(self.selected_features)
            print(f"自动选择显著特征（p<0.05），共{k}个")
        else:
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_scored = self.feature_selector.fit_transform(X, y)  # 这里已经是筛选后的k个特征
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()

        # 关键修改：直接使用X_scored（已筛选），无需再用掩码索引
        X_selected = pd.DataFrame(
            X_scored,  # 移除 [:, self.feature_selector.get_support()]
            columns=self.selected_features
        )

        # 输出特征重要性排序（保持不变）
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'f_score': self.feature_selector.scores_,
            'p_value': self.feature_selector.pvalues_
        }).sort_values('f_score', ascending=False)

        print("Top10重要特征：")
        print(feature_scores.head(10)[['feature', 'f_score', 'p_value']].round(3))

        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的选择器转换特征"""
        if self.feature_selector is None:
            raise ValueError("特征选择器尚未拟合，请先调用select_features方法")
        X_selected = self.feature_selector.transform(X)
        return pd.DataFrame(X_selected, columns=self.selected_features)

    # def transform_base(self, df):
    #     """
    #     仅执行基础特征工程（不进行特征选择）
    #     用于训练流程的前半部分
    #     """
    #     df = df.copy()
    #
    #     # 清洗特征
    #     df = self.clean_feature(df)
    #
    #     # 添加自定义特征
    #     df = self.add_custom_features(df)
    #
    #     return df


# 测试代码
if __name__ == "__main__":
    from loader import DataLoader
    from preprocessor import DataPreprocessor

    # 加载数据
    loader = DataLoader()
    df = loader.load_data(data_type='categorical')
    train_df, test_df = loader.split_train_test(df)

    # 1. 特征工程：创建新特征
    fe = FeatureEngineer()
    train_df_with_new_features = fe.create_features(train_df, is_numeric_data=False)
    test_df_with_new_features = fe.create_features(test_df, is_numeric_data=False)

    # 2. 预处理（分离特征和目标变量）
    preprocessor = DataPreprocessor()
    # 拟合预处理管道（基于新增特征后的训练集）
    X_train_raw = train_df_with_new_features.drop(columns=['default'])
    y_train = train_df_with_new_features['default']
    preprocessor._identify_feature_types(train_df_with_new_features)
    preprocessor.build_pipeline()
    X_train_processed = preprocessor.pipeline.fit_transform(X_train_raw)
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=preprocessor.feature_names_out)

    # 3. 特征选择
    X_train_selected = fe.select_features(X_train_processed_df, y_train, k=15)
    print(f"选择后训练集特征形状：{X_train_selected.shape}")

    # 转换测试集
    X_test_raw = test_df_with_new_features.drop(columns=['default'])
    X_test_processed = preprocessor.pipeline.transform(X_test_raw)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=preprocessor.feature_names_out)
    X_test_selected = fe.transform(X_test_processed_df)
    print(f"选择后测试集特征形状：{X_test_selected.shape}")