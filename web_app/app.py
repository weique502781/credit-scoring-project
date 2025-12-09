import os
from flask import Flask, render_template, send_from_directory
import shutil

app = Flask(__name__)

# 配置路径
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
STATIC_REPORTS_DIR = os.path.join(app.root_path, 'static', 'reports')
STATIC_IMAGES_DIR = os.path.join(app.root_path, 'static', 'images')


def setup_static_files():
    """确保静态文件目录存在并复制报告文件"""
    # 创建静态目录
    os.makedirs(STATIC_REPORTS_DIR, exist_ok=True)
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

    # 创建占位图片（用于图片加载失败时）
    placeholder_path = os.path.join(STATIC_IMAGES_DIR, 'placeholder.png')
    if not os.path.exists(placeholder_path):
        with open(placeholder_path, 'w') as f:
            f.write('')  # 空文件，实际部署时应替换为真实占位图片

    # 复制报告文件到静态目录
    if os.path.exists(REPORTS_DIR):
        for item in os.listdir(REPORTS_DIR):
            src = os.path.join(REPORTS_DIR, item)
            dest = os.path.join(STATIC_REPORTS_DIR, item)

            # 复制文件或目录
            if os.path.isfile(src):
                shutil.copy2(src, dest)
            elif os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data-analysis')
def data_analysis():
    return render_template('data-analysis.html')


@app.route('/model-results')
def model_results():
    return render_template('model-results.html')


@app.route('/model-comparison')
def model_comparison():
    return render_template('model-comparison.html')


@app.route('/static/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(STATIC_REPORTS_DIR, filename)


if __name__ == '__main__':
    # 初始化静态文件
    setup_static_files()
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)