import pandas as pd
import os
from typing import Optional, Tuple


class DataLoader:
    """德国信用数据集加载器，支持加载原始分类格式或数值格式数据"""

    # 按german.doc定义的20个特征名称
    FEATURE_NAMES = [
        'status_checking_account',  # 特征1：现有支票账户状态（分类）
        'duration',  # 特征2：贷款期限（月，数值）
        'credit_history',  # 特征3：信用历史（分类）
        'purpose',  # 特征4：贷款目的（分类）
        'credit_amount',  # 特征5：贷款金额（数值）
        'savings_account',  # 特征6：储蓄账户/债券（分类）
        'present_employment',  # 特征7：当前就业年限（分类）
        'installment_rate',  # 特征8：月供占可支配收入比例（数值）
        'personal_status_sex',  # 特征9：个人状态和性别（分类）
        'other_debtors',  # 特征10：其他债务人/担保人（分类）
        'present_residence',  # 特征11：当前居住年限（数值）
        'property',  # 特征12：资产（分类）
        'age',  # 特征13：年龄（岁，数值）
        'other_installment_plans',  # 特征14：其他分期付款计划（分类）
        'housing',  # 特征15：住房类型（分类）
        'existing_credits',  # 特征16：本银行现有贷款数（数值）
        'job',  # 特征17：职业（分类）
        'num_maintenance_people',  # 特征18：需赡养人数（数值）
        'telephone',  # 特征19：是否有电话（分类）
        'foreign_worker'  # 特征20：是否外籍工人（分类）
    ]
    TARGET_COL = 'default'  # 目标列：是否违约（0=Good，1=Bad）

    def __init__(self):
        # 关键修改：获取项目根目录（src的上级目录的上级目录）
        # 当前文件路径是 src/data/loader.py，所以向上两级到项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # 拼接 data/raw/ 的绝对路径
        self.data_dir = os.path.join(self.project_root, 'data', 'raw')
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self, data_type: str = 'categorical', target_map: dict = {1: 0, 2: 1}) -> pd.DataFrame:
        """
        加载德国信用数据集
        :param data_type: 数据类型，'categorical'加载german.data（含分类特征），'numeric'加载german.data-numeric（全数值）
        :param target_map: 目标变量映射，默认1→0（Good）、2→1（Bad）
        :return: 带列名的DataFrame（含特征+目标列）
        """
        # 确定数据文件路径
        if data_type == 'categorical':
            file_path = os.path.join(self.data_dir, 'german.data')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件 {file_path} 不存在，请从UCI仓库下载")
            # 加载categorical数据（空格分隔，无列名）
            df = pd.read_csv(file_path, sep='\s+', header=None)
            # 前20列为特征，最后1列为目标变量
            df.columns = self.FEATURE_NAMES + [self.TARGET_COL]

        elif data_type == 'numeric':
            file_path = os.path.join(self.data_dir, 'german.data-numeric')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件 {file_path} 不存在，请从UCI仓库下载")
            # 加载numeric数据（空格分隔，无列名，前24列为特征，最后1列为目标）
            df = pd.read_csv(file_path, sep='\s+', header=None)
            # 数值型数据特征列名（按原文档说明为24个数值特征）
            numeric_feature_names = [f'feature_{i + 1}' for i in range(24)]
            df.columns = numeric_feature_names + [self.TARGET_COL]

        else:
            raise ValueError("data_type必须为'categorical'或'numeric'")

        # 映射目标变量
        df[self.TARGET_COL] = df[self.TARGET_COL].map(target_map)
        # 检查目标变量映射是否成功
        if df[self.TARGET_COL].isnull().any():
            raise ValueError(f"目标变量存在未映射的值，原始值：{df[self.TARGET_COL].unique()}")

        print(f"成功加载{data_type}类型数据，形状：{df.shape}")
        print(f"违约分布：\n{df[self.TARGET_COL].value_counts(normalize=True).round(3)}")
        return df

    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        划分训练集和测试集
        :param df: 完整数据集
        :param test_size: 测试集比例
        :param random_state: 随机种子（保证可复现）
        :return: (训练集, 测试集)
        """
        train_df = df.sample(frac=1 - test_size, random_state=random_state)
        test_df = df.drop(train_df.index)
        print(f"训练集形状：{train_df.shape}，测试集形状：{test_df.shape}")
        return train_df, test_df


# 测试代码（运行时验证）
if __name__ == "__main__":
    loader = DataLoader()
    # 加载含分类特征的数据集
    df = loader.load_data(data_type='categorical')
    print("\n前5行数据：")
    print(df.head())

    # 划分训练集和测试集
    train_df, test_df = loader.split_train_test(df)

    # 关键修改：基于项目根目录拼接保存路径（根目录/data/raw/）
    train_save_path = os.path.join(loader.data_dir, 'train_raw.csv')  # loader.data_dir 已指向根目录/data/raw/
    test_save_path = os.path.join(loader.data_dir, 'test_raw.csv')

    # 保存训练集和测试集到正确位置
    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    print(f"\n训练集已保存至: {train_save_path}")
    print(f"测试集已保存至: {test_save_path}")