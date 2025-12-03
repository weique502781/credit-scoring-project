from flask import Flask, render_template, request, jsonify, send_file, session
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
from datetime import datetime
from werkzeug.utils import secure_filename
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interpretability.explainer import ModelExplainer
from src.interpretability.rules_extractor import DecisionRulesExtractor

app = Flask(__name__)
app.secret_key = 'credit_scoring_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# 全局变量
MODELS = {}
PREPROCESSOR = None
FEATURE_NAMES = []
EXPLAINER = None
RULES_EXTRACTOR = None

# 数据集信息（德国信用数据集）
DATASET_INFO = {
    'features': {
        'checking_account': {
            'name': 'Checking Account Status',
            'description': 'Status of existing checking account',
            'categories': {
                'A11': '< 0 DM',
                'A12': '0 <= ... < 200 DM',
                'A13': '>= 200 DM',
                'A14': 'no checking account'
            }
        },
        'duration': {
            'name': 'Loan Duration',
            'description': 'Duration in months',
            'type': 'numeric',
            'range': [4, 72]
        },
        'credit_history': {
            'name': 'Credit History',
            'description': 'Previous credit history',
            'categories': {
                'A30': 'no credits taken/all credits paid back duly',
                'A31': 'all credits at this bank paid back duly',
                'A32': 'existing credits paid back duly till now',
                'A33': 'delay in paying off in the past',
                'A34': 'critical account/other credits existing'
            }
        },
        'purpose': {
            'name': 'Purpose',
            'description': 'Purpose of the loan',
            'categories': {
                'A40': 'car (new)',
                'A41': 'car (used)',
                'A42': 'furniture/equipment',
                'A43': 'radio/television',
                'A44': 'domestic appliances',
                'A45': 'repairs',
                'A46': 'education',
                'A47': 'vacation',
                'A48': 'retraining',
                'A49': 'business',
                'A410': 'others'
            }
        },
        'amount': {
            'name': 'Credit Amount',
            'description': 'Credit amount in DM',
            'type': 'numeric',
            'range': [250, 18424]
        },
        'savings': {
            'name': 'Savings Account/Bonds',
            'description': 'Savings account/bonds',
            'categories': {
                'A61': '< 100 DM',
                'A62': '100 <= ... < 500 DM',
                'A63': '500 <= ... < 1000 DM',
                'A64': '>= 1000 DM',
                'A65': 'unknown/no savings account'
            }
        },
        'employment': {
            'name': 'Present Employment Since',
            'description': 'Present employment since',
            'categories': {
                'A71': 'unemployed',
                'A72': '< 1 year',
                'A73': '1 <= ... < 4 years',
                'A74': '4 <= ... < 7 years',
                'A75': '>= 7 years'
            }
        },
        'installment_rate': {
            'name': 'Installment Rate',
            'description': 'Installment rate in percentage of disposable income',
            'type': 'numeric',
            'range': [1, 4]
        },
        'personal_status': {
            'name': 'Personal Status and Sex',
            'description': 'Personal status and sex',
            'categories': {
                'A91': 'male: divorced/separated',
                'A92': 'female: divorced/separated/married',
                'A93': 'male: single',
                'A94': 'male: married/widowed',
                'A95': 'female: single'
            }
        },
        'debtors': {
            'name': 'Other Debtors / Guarantors',
            'description': 'Other debtors / guarantors',
            'categories': {
                'A101': 'none',
                'A102': 'co-applicant',
                'A103': 'guarantor'
            }
        },
        'residence': {
            'name': 'Present Residence Since',
            'description': 'Present residence since (years)',
            'type': 'numeric',
            'range': [1, 4]
        },
        'property': {
            'name': 'Property',
            'description': 'Property',
            'categories': {
                'A121': 'real estate',
                'A122': 'building society savings agreement/life insurance',
                'A123': 'car or other',
                'A124': 'unknown / no property'
            }
        },
        'age': {
            'name': 'Age',
            'description': 'Age in years',
            'type': 'numeric',
            'range': [19, 75]
        },
        'other_plans': {
            'name': 'Other Installment Plans',
            'description': 'Other installment plans',
            'categories': {
                'A141': 'bank',
                'A142': 'stores',
                'A143': 'none'
            }
        },
        'housing': {
            'name': 'Housing',
            'description': 'Housing',
            'categories': {
                'A151': 'rent',
                'A152': 'own',
                'A153': 'for free'
            }
        },
        'existing_credits': {
            'name': 'Number of Existing Credits',
            'description': 'Number of existing credits at this bank',
            'type': 'numeric',
            'range': [1, 4]
        },
        'job': {
            'name': 'Job',
            'description': 'Job type',
            'categories': {
                'A171': 'unemployed/unskilled - non-resident',
                'A172': 'unskilled - resident',
                'A173': 'skilled employee/official',
                'A174': 'management/self-employed/highly qualified employee/officer'
            }
        },
        'people_liable': {
            'name': 'Number of People Liable',
            'description': 'Number of people being liable to provide maintenance for',
            'type': 'numeric',
            'range': [1, 2]
        },
        'telephone': {
            'name': 'Telephone',
            'description': 'Telephone',
            'categories': {
                'A191': 'none',
                'A192': 'yes, registered under the customers name'
            }
        },
        'foreign_worker': {
            'name': 'Foreign Worker',
            'description': 'Foreign worker',
            'categories': {
                'A201': 'yes',
                'A202': 'no'
            }
        }
    }
}


def load_models():
    """加载所有训练好的模型"""
    global MODELS, PREPROCESSOR, FEATURE_NAMES, EXPLAINER, RULES_EXTRACTOR

    print("Loading models and preprocessing pipeline...")

    try:
        # 加载预处理管道
        if os.path.exists('models/preprocessor.pkl'):
            PREPROCESSOR = joblib.load('models/preprocessor.pkl')
            if hasattr(PREPROCESSOR, 'feature_names'):
                FEATURE_NAMES = PREPROCESSOR.feature_names
            else:
                # 从配置或数据集中获取特征名称
                FEATURE_NAMES = list(DATASET_INFO['features'].keys())

        # 加载基模型
        model_files = {
            'logistic_regression': 'models/saved_models/logistic_regression.pkl',
            'decision_tree': 'models/saved_models/decision_tree.pkl',
            'svm': 'models/saved_models/svm.pkl',
            'naive_bayes': 'models/saved_models/naive_bayes.pkl',
            'custom_adaboost': 'models/saved_models/ensemble/custom_adaboost.pkl',
            'sklearn_adaboost': 'models/saved_models/ensemble/sklearn_adaboost.pkl'
        }

        loaded_count = 0
        for name, path in model_files.items():
            if os.path.exists(path):
                try:
                    MODELS[name] = {
                        'model': joblib.load(path),
                        'type': name
                    }
                    loaded_count += 1
                    print(f"  ✓ Loaded: {name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {name}: {e}")

        # 初始化解释器
        if FEATURE_NAMES:
            EXPLAINER = ModelExplainer(feature_names=FEATURE_NAMES)
            RULES_EXTRACTOR = DecisionRulesExtractor(feature_names=FEATURE_NAMES)
            print(f"  ✓ Initialized explainers")

        print(f"Successfully loaded {loaded_count} models")

        # 加载评估结果
        load_evaluation_results()

    except Exception as e:
        print(f"Error loading models: {e}")
        MODELS = {}


def load_evaluation_results():
    """加载评估结果"""
    global EVALUATION_RESULTS

    EVALUATION_RESULTS = {}

    try:
        # 加载模型对比结果
        if os.path.exists('reports/model_comparison.csv'):
            EVALUATION_RESULTS['comparison'] = pd.read_csv('reports/model_comparison.csv').to_dict('records')

        # 加载ROC分析结果
        if os.path.exists('reports/roc_analysis_report.csv'):
            EVALUATION_RESULTS['roc'] = pd.read_csv('reports/roc_analysis_report.csv').to_dict('records')

        # 加载特征重要性
        if os.path.exists('reports/feature_importance.csv'):
            EVALUATION_RESULTS['feature_importance'] = pd.read_csv('reports/feature_importance.csv').to_dict('records')

        print("Evaluation results loaded")

    except Exception as e:
        print(f"Error loading evaluation results: {e}")


def create_sample_instance():
    """创建示例实例"""
    sample = {
        'checking_account': 'A14',
        'duration': '24',
        'credit_history': 'A34',
        'purpose': 'A43',
        'amount': '5000',
        'savings': 'A65',
        'employment': 'A75',
        'installment_rate': '2',
        'personal_status': 'A93',
        'debtors': 'A101',
        'residence': '2',
        'property': 'A121',
        'age': '35',
        'other_plans': 'A143',
        'housing': 'A152',
        'existing_credits': '1',
        'job': 'A173',
        'people_liable': '1',
        'telephone': 'A192',
        'foreign_worker': 'A201'
    }
    return sample


# 在应用启动时加载模型
load_models()


@app.route('/')
def home():
    """首页"""
    model_info = []
    for name, info in MODELS.items():
        model_info.append({
            'name': name.replace('_', ' ').title(),
            'type': info['type'],
            'loaded': True
        })

    # 获取系统状态
    system_status = {
        'total_models': len(MODELS),
        'preprocessor_loaded': PREPROCESSOR is not None,
        'explainer_loaded': EXPLAINER is not None,
        'evaluation_loaded': 'comparison' in EVALUATION_RESULTS
    }

    return render_template('index.html',
                           models=model_info,
                           system_status=system_status,
                           dataset_info=DATASET_INFO)


@app.route('/predict', methods=['GET'])
def predict_form():
    """预测表单页面"""
    sample_instance = create_sample_instance()

    return render_template('predict.html',
                           sample_instance=sample_instance,
                           dataset_info=DATASET_INFO,
                           models=list(MODELS.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    try:
        # 获取表单数据
        form_data = request.form.to_dict()
        model_name = form_data.get('model', 'custom_adaboost')

        # 验证模型
        if model_name not in MODELS:
            return render_template('error.html',
                                   error_message=f"Model '{model_name}' not found")

        # 准备数据
        input_df = pd.DataFrame([form_data])

        # 保存原始数据用于展示
        original_values = {}
        for feature, info in DATASET_INFO['features'].items():
            if feature in form_data:
                value = form_data[feature]
                if 'categories' in info and value in info['categories']:
                    original_values[feature] = {
                        'code': value,
                        'description': info['categories'][value]
                    }
                else:
                    original_values[feature] = {
                        'code': value,
                        'description': value
                    }

        # 预处理
        if PREPROCESSOR is None:
            return render_template('error.html',
                                   error_message="Preprocessor not loaded")

        try:
            processed_input = PREPROCESSOR.transform(input_df)
        except Exception as e:
            return render_template('error.html',
                                   error_message=f"Data preprocessing failed: {str(e)}")

        # 进行预测
        model = MODELS[model_name]['model']

        try:
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(processed_input)[0]
                prediction = model.predict(processed_input)[0]

                # 计算置信度
                confidence = max(probability)
                predicted_class_idx = np.argmax(probability)

                result = {
                    'prediction': 'Good Credit' if prediction == 1 else 'Bad Credit',
                    'prediction_numeric': int(prediction),
                    'probability_good': f"{probability[1]:.2%}",
                    'probability_bad': f"{probability[0]:.2%}",
                    'probability_good_raw': float(probability[1]),
                    'probability_bad_raw': float(probability[0]),
                    'confidence': f"{confidence:.2%}",
                    'model_used': model_name.replace('_', ' ').title(),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_type': type(model).__name__
                }
            else:
                prediction = model.predict(processed_input)[0]
                result = {
                    'prediction': 'Good Credit' if prediction == 1 else 'Bad Credit',
                    'prediction_numeric': int(prediction),
                    'model_used': model_name.replace('_', ' ').title(),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_type': type(model).__name__
                }

            # 将结果保存到session用于后续解释
            session['last_prediction'] = {
                'form_data': form_data,
                'result': result,
                'processed_input': processed_input.tolist(),
                'model_name': model_name
            }

            return render_template('predict_result.html',
                                   result=result,
                                   original_values=original_values,
                                   dataset_info=DATASET_INFO)

        except Exception as e:
            return render_template('error.html',
                                   error_message=f"Prediction failed: {str(e)}")

    except Exception as e:
        return render_template('error.html',
                               error_message=f"Error processing request: {str(e)}")


@app.route('/explain', methods=['GET'])
def explain_home():
    """可解释性主页"""
    # 获取可用的解释方法
    explanation_methods = [
        {'id': 'feature_importance', 'name': 'Feature Importance', 'description': 'Global feature importance analysis'},
        {'id': 'decision_rules', 'name': 'Decision Rules',
         'description': 'Extract human-readable rules from decision trees'},
        {'id': 'lime', 'name': 'LIME Explanation', 'description': 'Local interpretable model-agnostic explanations'},
        {'id': 'shap', 'name': 'SHAP Values', 'description': 'SHapley Additive exPlanations'}
    ]

    # 检查是否保存了上次预测
    has_last_prediction = 'last_prediction' in session

    return render_template('explain_home.html',
                           explanation_methods=explanation_methods,
                           has_last_prediction=has_last_prediction)


@app.route('/explain/<method>', methods=['GET'])
def explain_method(method):
    """特定解释方法页面"""
    if method == 'feature_importance':
        return explain_feature_importance()
    elif method == 'decision_rules':
        return explain_decision_rules()
    elif method == 'lime':
        return explain_lime()
    elif method == 'shap':
        return explain_shap()
    else:
        return render_template('error.html',
                               error_message=f"Unknown explanation method: {method}")


def explain_feature_importance():
    """特征重要性解释"""
    try:
        # 尝试加载特征重要性数据
        feature_importance_data = None
        if os.path.exists('reports/feature_importance.csv'):
            df = pd.read_csv('reports/feature_importance.csv')
            feature_importance_data = df.to_dict('records')

        # 为每个模型创建特征重要性图
        importance_plots = {}

        for model_name, model_info in MODELS.items():
            model = model_info['model']

            # 检查模型是否有特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                # 创建DataFrame
                importance_df = pd.DataFrame({
                    'feature': FEATURE_NAMES[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False).head(15)

                # 创建Plotly图表
                fig = px.bar(importance_df,
                             x='importance',
                             y='feature',
                             orientation='h',
                             title=f'{model_name.replace("_", " ").title()} - Feature Importance',
                             labels={'importance': 'Importance', 'feature': 'Feature'})

                fig.update_layout(height=600, showlegend=False)

                # 转换为HTML
                plot_html = plotly.io.to_html(fig, full_html=False)
                importance_plots[model_name] = plot_html

        return render_template('explain_feature_importance.html',
                               feature_importance_data=feature_importance_data,
                               importance_plots=importance_plots,
                               total_models=len(MODELS))

    except Exception as e:
        return render_template('error.html',
                               error_message=f"Feature importance analysis failed: {str(e)}")


def explain_decision_rules():
    """决策规则解释"""
    try:
        # 检查是否有决策树模型
        tree_models = {}
        for model_name, model_info in MODELS.items():
            model = model_info['model']
            if hasattr(model, 'tree_') or 'tree' in model_name.lower():
                tree_models[model_name] = model

        if not tree_models:
            return render_template('explain_decision_rules.html',
                                   no_tree_models=True)

        # 提取规则
        all_rules = {}

        for model_name, model in tree_models.items():
            try:
                if RULES_EXTRACTOR:
                    if hasattr(model, 'tree_'):  # 单个决策树
                        rules_df = RULES_EXTRACTOR.extract_rules_from_tree(model, max_depth=4)
                        simplified_rules = RULES_EXTRACTOR.simplify_rules(rules_df)

                        all_rules[model_name] = {
                            'type': 'single_tree',
                            'total_rules': len(rules_df),
                            'simplified_rules': simplified_rules.head(20).to_dict('records'),
                            'has_more': len(rules_df) > 20
                        }

                    elif hasattr(model, 'estimators_'):  # AdaBoost
                        rules_dict = RULES_EXTRACTOR.extract_rules_from_adaboost(
                            model, max_trees=5, max_depth=2
                        )

                        all_rules[model_name] = {
                            'type': 'adaboost',
                            'total_trees': len(model.estimators_),
                            'trees_analyzed': len(rules_dict),
                            'rules': rules_dict
                        }

            except Exception as e:
                print(f"Error extracting rules from {model_name}: {e}")
                all_rules[model_name] = {
                    'type': 'error',
                    'error': str(e)
                }

        # 生成报告
        report_html = None
        if all_rules and RULES_EXTRACTOR:
            try:
                # 使用第一个树的规则生成报告
                for model_name, rules_info in all_rules.items():
                    if rules_info['type'] == 'single_tree' and 'simplified_rules' in rules_info:
                        if rules_info['simplified_rules']:
                            rules_df = pd.DataFrame(rules_info['simplified_rules'])
                            report = RULES_EXTRACTOR.generate_rule_report(rules_df)
                            report_html = report.replace('\n', '<br>').replace('|', ' | ')
                            break
            except Exception as e:
                print(f"Error generating report: {e}")

        return render_template('explain_decision_rules.html',
                               tree_models=tree_models,
                               all_rules=all_rules,
                               report_html=report_html,
                               no_tree_models=False)

    except Exception as e:
        return render_template('error.html',
                               error_message=f"Decision rules extraction failed: {str(e)}")


def explain_lime():
    """LIME解释"""
    try:
        # 检查是否有上次预测
        if 'last_prediction' not in session:
            return render_template('explain_lime.html',
                                   no_prediction=True)

        prediction_data = session['last_prediction']
        model_name = prediction_data['model_name']

        if model_name not in MODELS:
            return render_template('error.html',
                                   error_message=f"Model '{model_name}' not found")

        model = MODELS[model_name]['model']
        processed_input = np.array(prediction_data['processed_input'])

        # 需要训练数据来初始化LIME，这里使用示例数据
        # 在实际应用中，应该加载训练数据
        from sklearn.datasets import make_classification
        X_sample, _ = make_classification(n_samples=100, n_features=processed_input.shape[1],
                                          random_state=42)

        # 初始化LIME解释器
        if EXPLAINER:
            EXPLAINER.initialize_lime_explainer(X_sample)

            # 生成解释
            explanation = EXPLAINER.explain_with_lime(
                model, processed_input[0],
                num_features=10, show_plot=False,
                save_path='static/plots/lime_explanation.png'
            )

            if explanation:
                # 准备可视化数据
                features = []
                weights = []

                for feat, weight in explanation['explanation']:
                    feature_name = feat.split(' ')