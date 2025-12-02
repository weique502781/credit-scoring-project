from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import sys

sys.path.append('..')

from src.data.preprocessor import DataPreprocessor
from src.interpretability.explainer import ModelExplainer

app = Flask(__name__)

# 全局变量
MODELS = {}
PREPROCESSOR = None
EXPLAINER = None
FEATURE_NAMES = []


def load_models():
    """加载所有训练好的模型"""
    global MODELS, PREPROCESSOR, EXPLAINER, FEATURE_NAMES

    # 加载预处理管道
    if os.path.exists('models/preprocessor.pkl'):
        PREPROCESSOR = joblib.load('models/preprocessor.pkl')
        FEATURE_NAMES = PREPROCESSOR.feature_names if hasattr(PREPROCESSOR, 'feature_names') else []

    # 加载基模型
    model_files = {
        'logistic_regression': 'models/saved_models/logistic_regression.pkl',
        'decision_tree': 'models/saved_models/decision_tree.pkl',
        'svm': 'models/saved_models/svm.pkl',
        'naive_bayes': 'models/saved_models/naive_bayes.pkl',
        'custom_adaboost': 'models/saved_models/ensemble/custom_adaboost.pkl',
        'sklearn_adaboost': 'models/saved_models/ensemble/sklearn_adaboost.pkl'
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            MODELS[name] = joblib.load(path)
            print(f"Loaded model: {name}")

    # 创建解释器
    if FEATURE_NAMES:
        EXPLAINER = ModelExplainer(feature_names=FEATURE_NAMES)

    print(f"Loaded {len(MODELS)} models successfully!")


# 在应用启动时加载模型
load_models()


@app.route('/')
def home():
    """首页"""
    model_info = []
    for name, model in MODELS.items():
        model_info.append({
            'name': name,
            'type': type(model).__name__
        })

    return render_template('index.html',
                           models=model_info,
                           total_models=len(model_info))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """预测页面"""
    if request.method == 'GET':
        # 显示预测表单
        feature_examples = {
            'duration': {'min': 4, 'max': 72, 'default': 24},
            'amount': {'min': 250, 'max': 18424, 'default': 5000},
            'age': {'min': 19, 'max': 75, 'default': 35},
            'installment_rate': {'min': 1, 'max': 4, 'default': 2}
        }

        categorical_options = {
            'checking_account': ['A11', 'A12', 'A13', 'A14'],
            'credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
            'purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
            'savings': ['A61', 'A62', 'A63', 'A64', 'A65']
        }

        return render_template('predict.html',
                               feature_examples=feature_examples,
                               categorical_options=categorical_options)

    else:
        # 处理预测请求
        try:
            # 获取表单数据
            form_data = request.form.to_dict()

            # 转换为DataFrame
            input_df = pd.DataFrame([form_data])

            # 预处理输入
            if PREPROCESSOR is not None:
                processed_input = PREPROCESSOR.transform(input_df)
            else:
                processed_input = input_df.values

            # 选择模型
            model_name = form_data.get('model', 'custom_adaboost')
            if model_name not in MODELS:
                model_name = 'custom_adaboost'

            model = MODELS[model_name]

            # 进行预测
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(processed_input)[0]
                prediction = model.predict(processed_input)[0]

                # 转换为可读结果
                result = {
                    'prediction': 'Good Credit' if prediction == 1 else 'Bad Credit',
                    'probability_good': f"{probability[1]:.2%}",
                    'probability_bad': f"{probability[0]:.2%}",
                    'model_used': model_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                prediction = model.predict(processed_input)[0]
                result = {
                    'prediction': 'Good Credit' if prediction == 1 else 'Bad Credit',
                    'model_used': model_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

            return render_template('predict_result.html', result=result)

        except Exception as e:
            return render_template('error.html',
                                   error_message=f"预测失败: {str(e)}")


@app.route('/explain', methods=['GET', 'POST'])
def explain():
    """模型解释页面"""
    if request.method == 'GET':
        return render_template('explain.html')

    else:
        try:
            # 获取要解释的样本索引
            sample_idx = int(request.form.get('sample_idx', 0))

            # 这里需要加载测试数据
            # 为了简化，我们假设有测试数据可用
            # 在实际应用中，你需要从文件加载测试数据

            # 示例：使用第一个模型进行解释
            model_name = list(MODELS.keys())[0]
            model = MODELS[model_name]

            # 获取示例数据（这里需要实际数据）
            # X_test_sample = X_test[sample_idx:sample_idx+1]

            # 使用解释器
            if EXPLAINER is not None:
                # 这里需要实际数据
                # explanation = EXPLAINER.lime_explain(model, X_train, X_test_sample[0])
                explanation = None

                return render_template('explain_result.html',
                                       explanation_available=True,
                                       sample_idx=sample_idx,
                                       model_name=model_name)
            else:
                return render_template('explain_result.html',
                                       explanation_available=False)

        except Exception as e:
            return render_template('error.html',
                                   error_message=f"解释失败: {str(e)}")


@app.route('/compare', methods=['GET'])
def compare():
    """模型对比页面"""
    try:
        # 加载比较结果
        if os.path.exists('reports/model_comparison.csv'):
            comparison_df = pd.read_csv('reports/model_comparison.csv')
            comparison_data = comparison_df.to_dict('records')
        else:
            comparison_data = []

        # 加载ROC数据
        roc_data = {}
        if os.path.exists('reports/roc_analysis_report.csv'):
            roc_df = pd.read_csv('reports/roc_analysis_report.csv')
            roc_data = roc_df.to_dict('records')

        return render_template('compare.html',
                               comparison_data=comparison_data,
                               roc_data=roc_data,
                               total_models=len(comparison_data))

    except Exception as e:
        return render_template('error.html',
                               error_message=f"加载对比数据失败: {str(e)}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API预测接口"""
    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        # 处理输入
        input_df = pd.DataFrame([data['features']])

        # 预处理
        if PREPROCESSOR is not None:
            processed_input = PREPROCESSOR.transform(input_df)
        else:
            processed_input = input_df.values

        # 选择模型
        model_name = data.get('model', 'custom_adaboost')
        if model_name not in MODELS:
            model_name = 'custom_adaboost'

        model = MODELS[model_name]

        # 预测
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(processed_input)[0]
            prediction = model.predict(processed_input)[0]

            response = {
                'prediction': int(prediction),
                'probability_good': float(probability[1]),
                'probability_bad': float(probability[0]),
                'model_used': model_name
            }
        else:
            prediction = model.predict(processed_input)[0]
            response = {
                'prediction': int(prediction),
                'model_used': model_name
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def api_get_models():
    """获取可用模型列表"""
    model_list = [{'name': name, 'type': type(model).__name__}
                  for name, model in MODELS.items()]
    return jsonify({'models': model_list})


@app.route('/download/<filename>')
def download_file(filename):
    """下载文件"""
    file_path = f'reports/{filename}'

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return render_template('error.html',
                               error_message=f"文件 {filename} 不存在")


if __name__ == '__main__':
    # 确保必要的目录存在
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('models/saved_models/ensemble', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # 运行应用
    app.run(host='127.0.0.1', port=5000, debug=True)