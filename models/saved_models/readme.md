1. 模型文件目录
models/saved_models/
├── preprocessor.pkl # 数据预处理器（必需！）
├── feature_names.npy # 特征名称列表
├── train_config.json # 训练配置信息
├── feature_importance.csv # 特征重要性排名
├── logistic_regression.pkl # 逻辑回归模型
├── decision_tree.pkl # 决策树模型
├── svm.pkl # SVM模型
├── naive_bayes.pkl # 朴素贝叶斯模型
└── ensemble/ # 集成模型目录
├── custom_adaboost.pkl # AdaBoost
└── sklearn_adaboost.pkl # sklearn官方AdaBoost

2. 源代码目录
src/models/
├── init.py
├── base_models.py # 4种基分类器实现
├── adaboost_custom.py # AdaBoost实现
├── ensemble.py # 集成训练器
├── model_pipeline.py # 训练流水线
└── ensemble_strategy.py # 集成策略（硬/软投票）

3. 接口规范
class BaseModel:
    def train(self, X, y):            # 训练模型
    def predict(self, X):             # 预测类别
    def predict_proba(self, X):       # 预测概率
    def get_feature_importances(self, feature_names):  # 获取特征重要性

4. 模型性能总结
模型	准确率	       F1分数	训练时间	备注
SVM (RBF核)	       0.775	0.678	中等	    最佳模型
AdaBoost	       0.770	0.672	较快	    性能接近SVM
决策树	           0.720	0.615	快	    可解释性强
朴素贝叶斯	       0.710	0.598	非常快	计算效率高
逻辑回归	           0.695	0.583	快	    线性模型基准
sklearn AdaBoost   0.690	0.575	中等	    对比基准