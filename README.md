# 集成学习驱动的可解释性信用评分模型

## 📋 项目概述

本项目是一个基于机器学习的企业/个人信用风险评估系统，旨在准确预测信用违约风险，并提供模型决策的可解释性分析。系统集成了多种机器学习算法，并使用Adaboost进行集成学习，最后通过Web界面提供交互式预测和解释功能。

### ✨ 核心特性
- **多模型集成**: 实现了4种基分类器 + 2种Adaboost集成方法
- **ROC分析**: 完整的ROC曲线对比与阈值分析
- **可解释性**: 集成LIME和SHAP进行模型解释
- **Web界面**: 基于Flask的交互式Web应用
- **完整评估**: 全面的模型性能评估与可视化

## 🏗️ 项目结构
```

credit_scoring_project/
├── README.md # 项目说明文档（本文件）
├── requirements.txt # Python依赖包列表
├── main.py # 项目主程序入口
│
├── data/ # 数据目录
│ ├── raw/ # 原始数据（德国信用数据集）
│ │ └── german_credit.csv # 原始数据集文件
│ └── processed/ # 处理后的数据（运行时生成）
│
├── notebooks/ # Jupyter Notebooks
│ └── exploratory_analysis.ipynb # 数据探索性分析
│
├── src/ # 源代码目录
│ ├── init.py # Python包初始化文件
│ ├── data/ # 数据处理模块
│ │ ├── init.py # 模块初始化
│ │ ├── loader.py # 数据加载器（同学A负责）
│ │ ├── preprocessor.py # 数据预处理（同学B负责）
│ │ └── feature_engineer.py # 特征工程（同学B负责）
│ │
│ ├── models/ # 模型实现模块
│ │ ├── init.py # 模块初始化
│ │ ├── base_models.py # 4种基分类器实现（同学A负责）
│ │ ├── adaboost_custom.py # 自定义Adaboost实现（同学A负责）
│ │ └── ensemble.py # 集成学习方法（同学A负责）
│ │
│ ├── evaluation/ # 模型评估模块
│ │ ├── init.py # 模块初始化
│ │ ├── metrics.py # 评估指标计算（同学B负责）
│ │ ├── roc_analysis.py # ROC曲线分析（同学B负责）
│ │ └── visualizer.py # 结果可视化（同学B负责）
│ │
│ ├── interpretability/ # 可解释性模块
│ │ ├── init.py # 模块初始化
│ │ ├── explainer.py # 模型解释器（同学C负责）
│ │ └── rules_extractor.py # 决策规则提取（同学C负责）
│ │
│ └── utils/ # 工具函数
│ ├── init.py # 模块初始化
│ └── helpers.py # 辅助函数（通用）
│
├── web_app/ # Web应用程序（同学C负责）
│ ├── app.py # Flask主应用文件
│ ├── templates/ # HTML模板文件
│ │ ├── index.html # 首页
│ │ ├── predict.html # 预测页面
│ │ ├── explain.html # 解释页面
│ │ ├── compare.html # 对比页面
│ │ └── error.html # 错误页面
│ └── static/ # 静态资源
│ ├── css/ # CSS样式文件
│ │ └── style.css # 主样式表
│ └── js/ # JavaScript文件
│ └── script.js # 前端脚本
│
├── config/ # 配置文件目录
│ └── config.yaml # 项目配置文件
│
├── models/ # 模型保存目录（运行时生成）
│ └── saved_models/ # 训练好的模型文件
│ ├── logistic_regression.pkl # 逻辑回归模型
│ ├── decision_tree.pkl # 决策树模型
│ ├── svm.pkl # SVM模型
│ ├── naive_bayes.pkl # 朴素贝叶斯模型
│ ├── ensemble/ # 集成模型目录
│ │ ├── custom_adaboost.pkl # 自定义Adaboost
│ │ └── sklearn_adaboost.pkl # sklearn Adaboost
│ └── preprocessor.pkl # 数据预处理管道
│
├── reports/ # 报告和可视化结果（运行时生成）
│ ├── roc_curves.png # ROC曲线图
│ ├── model_comparison.csv # 模型对比结果
│ ├── feature_importance.png # 特征重要性图
│ ├── confusion_matrices.png # 混淆矩阵图
│ ├── lime_explanation.html # LIME解释结果
│ └── threshold_analysis_adaboost.png # 阈值分析图
│
├── tests/ # 单元测试目录
│ ├── test_data.py # 数据模块测试
│ ├── test_models.py # 模型模块测试
│ └── test_evaluation.py # 评估模块测试
│
└── .gitignore # Git忽略文件配置
```

## 👥 三人分工说明

### 同学A：算法核心与模型实现
- **职责范围**：`src/models/` 目录下的所有文件
- **核心任务**：
  1. 实现4种基分类器（逻辑回归、决策树、SVM、朴素贝叶斯）
  2. 从零实现Adaboost算法（不使用sklearn内置版本）
  3. 模型参数调优与验证
  4. 处理类别不平衡问题
- **交付物**：
  - `base_models.py`：四个基分类器的完整实现
  - `adaboost_custom.py`：自定义Adaboost算法实现
  - `model_pipeline.py`：训练流水线代码
  - 训练好的模型文件（.pkl格式）

### 同学B：数据处理与评估可视化
- **职责范围**：`src/data/` 和 `src/evaluation/` 目录
- **核心任务**：
  1. 数据加载、清洗与预处理
  2. 特征工程与特征选择
  3. **ROC曲线绘制与对比分析**（重点）
  4. 模型评估指标计算与可视化
- **交付物**：
  - `data_preprocessor.py`：完整的数据预处理管道
  - `roc_analysis.py`：ROC曲线分析模块
  - `visualization.py`：所有可视化图表代码
  - 评估报告和可视化图表文件

### 同学C：可解释性分析与系统集成
- **职责范围**：`src/interpretability/` 和 `web_app/` 目录
- **核心任务**：
  1. 实现LIME和SHAP解释器
  2. 决策规则提取与可视化
  3. Flask Web应用开发
  4. 项目文档与演示材料准备
- **交付物**：
  - `model_explainer.py`：可解释性分析模块
  - `web_app/`：完整的Web应用程序
  - 项目文档、演示PPT和演示视频
  - 用户使用手册

## 🚀 快速开始

### 环境配置
```bash
# 1. 克隆项目
git clone https://github.com/weique502781/credit-scoring-project.git
cd credit_scoring_project

# 2. 创建虚拟环境（推荐）
conda create -n credit_scoring python=3.8
conda activate credit_scoring

# 3. 安装依赖
pip install -r requirements.txt