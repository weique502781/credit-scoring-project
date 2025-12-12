import os
from flask import Flask, render_template, send_from_directory
import shutil
from PIL import Image, ImageDraw, ImageFont
import traceback

app = Flask(__name__)

# 配置路径
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
STATIC_REPORTS_DIR = os.path.join(app.root_path, 'static', 'reports')
STATIC_IMAGES_DIR = os.path.join(app.root_path, 'static', 'images')


def create_placeholder_image(path, text="图片加载中..."):
    """创建占位图片"""
    try:
        img = Image.new('RGB', (400, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # 尝试使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        # 计算文本位置
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (400 - text_width) // 2
        y = (300 - text_height) // 2

        draw.text((x, y), text, fill=(100, 100, 100), font=font)
        img.save(path)
        return True
    except Exception as e:
        print(f"创建占位图片失败: {e}")
        return False


def setup_static_files():
    """确保静态文件目录存在并复制报告文件"""
    # 创建静态目录
    os.makedirs(STATIC_REPORTS_DIR, exist_ok=True)
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

    # 创建占位图片
    placeholder_path = os.path.join(STATIC_IMAGES_DIR, 'placeholder.png')
    if not os.path.exists(placeholder_path):
        if create_placeholder_image(placeholder_path):
            print(f"✓ 创建占位图片: {placeholder_path}")
        else:
            # 创建空文件作为备用
            with open(placeholder_path, 'wb') as f:
                f.write(b'')
            print(f"⚠ 创建空占位文件: {placeholder_path}")

    # 复制所有报告文件到静态目录
    if os.path.exists(REPORTS_DIR):
        print(f"正在复制报告文件从 {REPORTS_DIR} 到 {STATIC_REPORTS_DIR}")

        # 定义需要复制的文件类型
        supported_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.csv', '.json', '.md', '.txt']

        for item in os.listdir(REPORTS_DIR):
            src = os.path.join(REPORTS_DIR, item)

            # 如果是目录，递归复制
            if os.path.isdir(src):
                dest = os.path.join(STATIC_REPORTS_DIR, item)
                try:
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                    print(f"  ✓ 复制目录: {item}")
                except Exception as e:
                    print(f"  ✗ 复制目录失败 {item}: {e}")

            # 如果是文件且是支持的格式
            elif os.path.isfile(src):
                _, ext = os.path.splitext(item)
                if ext.lower() in supported_extensions:
                    dest = os.path.join(STATIC_REPORTS_DIR, item)
                    try:
                        shutil.copy2(src, dest)
                        print(f"  ✓ 复制文件: {item}")
                    except Exception as e:
                        print(f"  ✗ 复制文件失败 {item}: {e}")
                else:
                    print(f"  ⚠ 跳过不支持的格式: {item}")

        print("✓ 报告文件复制完成")

        # 检查必需的文件是否都存在
        required_files = [
            'default_distribution.png',
            'numeric_features_distribution.png',
            'feature_correlation.png',
            'feature_target_correlation.png',
            'categorical_features_default_rate.png',
            'account_status_credit_amount.png',
            'credit_amount_duration_interaction.png',
            'model_comparison.png',
            'confusion_matrices.png',
            'confusion_matrices_normalized.png',
            'roc_curves.png',
            'pr_curves.png',
            'feature_importance_adaboost.png',
            'threshold_analysis_adaboost.png',
            'model_comparison.csv',
            'decision_rules_report.md',
            'decision_rules.json'
        ]

        existing_files = os.listdir(STATIC_REPORTS_DIR)
        missing_files = [f for f in required_files if f not in existing_files]

        if missing_files:
            print(f"⚠ 缺少 {len(missing_files)} 个必需文件:")
            for f in missing_files[:5]:  # 只显示前5个
                print(f"    - {f}")
            if len(missing_files) > 5:
                print(f"    ... 还有 {len(missing_files) - 5} 个文件")
    else:
        print(f"⚠ 报告目录不存在: {REPORTS_DIR}")
        print("  请先运行 main.py 生成分析报告")


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


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="页面不存在"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="服务器内部错误"), 500


if __name__ == '__main__':
    try:
        # 初始化静态文件
        setup_static_files()
        # 启动Flask应用
        print("\n" + "=" * 60)
        print("德国信用评分Web应用")
        print("=" * 60)
        print("访问地址: http://127.0.0.1:5000")
        print("首页: http://127.0.0.1:5000/")
        print("数据分析页: http://127.0.0.1:5000/data-analysis")
        print("模型结果页: http://127.0.0.1:5000/model-results")
        print("模型对比页: http://127.0.0.1:5000/model-comparison")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器")
        print("=" * 60 + "\n")

        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"✗ 应用启动失败: {e}")
        traceback.print_exc()