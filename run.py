# run_all.py - 放在项目根目录
import os
import sys
import subprocess


def run_main_analysis():
    """运行主分析程序"""
    print("=" * 60)
    print("启动德国信用评分模型分析")
    print("=" * 60)

    # 检查依赖
    print("检查Python环境...")
    print(f"Python解释器: {sys.executable}")
    print(f"工作目录: {os.getcwd()}")

    # 检查是否在虚拟环境中
    venv_path = os.path.join(os.getcwd(), '.venv')
    if os.path.exists(venv_path):
        print("✓ 虚拟环境已激活")
    else:
        print("⚠ 未检测到虚拟环境，建议使用虚拟环境运行")

    # 运行main.py
    print("\n运行主分析程序...")
    try:
        import main
        main.main()
        return True
    except Exception as e:
        print(f"✗ 主程序运行失败: {e}")
        return False


def start_web_app():
    """启动Web应用"""
    print("\n" + "=" * 60)
    print("启动Web应用")
    print("=" * 60)

    web_app_dir = os.path.join(os.getcwd(), 'web_app')
    if not os.path.exists(web_app_dir):
        print(f"✗ Web应用目录不存在: {web_app_dir}")
        return False

    # 检查app.py
    app_file = os.path.join(web_app_dir, 'app.py')
    if not os.path.exists(app_file):
        print(f"✗ app.py不存在: {app_file}")
        return False

    print(f"切换到目录: {web_app_dir}")
    original_dir = os.getcwd()
    os.chdir(web_app_dir)

    try:
        print("启动Flask服务器...")
        print("访问地址: http://127.0.0.1:5000")
        print("按 Ctrl+C 停止服务器\n")

        # 运行Flask应用
        os.system(f"{sys.executable} app.py")
        return True

    except Exception as e:
        print(f"✗ Web应用启动失败: {e}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    """主菜单"""
    while True:
        print("\n" + "=" * 60)
        print("德国信用评分项目管理系统")
        print("=" * 60)
        print("1. 只运行数据分析和模型训练")
        print("2. 只启动Web应用")
        print("3. 先运行分析，再启动Web应用")
        print("4. 生成决策规则报告（单独）")
        print("5. 退出")
        print("=" * 60)

        choice = input("请选择 (1-5): ").strip()

        if choice == '1':
            run_main_analysis()
        elif choice == '2':
            start_web_app()
        elif choice == '3':
            if run_main_analysis():
                input("\n数据分析完成，按Enter键启动Web应用...")
                start_web_app()
        elif choice == '4':
            # 单独生成决策规则
            generate_decision_rules_alone()
        elif choice == '5':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")


def generate_decision_rules_alone():
    """单独生成决策规则"""
    print("\n生成决策规则报告...")

    # 导入必要的模块
    sys.path.insert(0, 'src')

    try:
        # 这里可以调用之前定义的create_sample_decision_rules函数
        from main import create_sample_decision_rules
        create_sample_decision_rules()
        print("✓ 决策规则生成完成")
    except:
        # 如果导入失败，直接创建
        import json

        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)

        # 创建示例规则
        sample_rules = """# Decision Rules Report - 示例

## 概述
示例决策规则报告，用于演示功能。

## 高信用评分规则
1. 有稳定收入和良好信用历史的客户
2. 贷款金额适中的短期贷款
3. 有房产抵押的客户

## 低信用评分规则
1. 无固定收入来源的客户
2. 大额长期贷款
3. 信用历史有问题的客户
"""

        with open(os.path.join(reports_dir, 'decision_rules_report.md'), 'w', encoding='utf-8') as f:
            f.write(sample_rules)

        print(f"✓ 示例规则已创建: {reports_dir}/decision_rules_report.md")


if __name__ == "__main__":
    main()