#!/bin/bash

# 高频金融数据分析应用启动脚本

echo "=========================================="
echo "  高频金融数据分析应用启动脚本"
echo "=========================================="
echo ""

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "警告: 未检测到虚拟环境"
    echo "建议在虚拟环境中运行应用"
    echo ""
    echo "如需创建虚拟环境:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate  # Linux/Mac"
    echo "  venv\\Scripts\\activate     # Windows"
    echo ""
    read -p "是否继续? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消启动"
        exit 1
    fi
else
    echo "检测到虚拟环境: $VIRTUAL_ENV"
fi

# 检查依赖包
echo ""
echo "检查依赖包..."
echo "如果缺少依赖包，请运行: pip install -r requirements.txt"
echo ""

# 尝试导入关键包
python3 -c "
try:
    import streamlit
    print('✓ Streamlit 已安装')
except ImportError:
    print('✗ Streamlit 未安装')

try:
    import pandas
    print('✓ Pandas 已安装')
except ImportError:
    print('✗ Pandas 未安装')

try:
    import plotly
    print('✓ Plotly 已安装')
except ImportError:
    print('✗ Plotly 未安装')

try:
    import numpy
    print('✓ NumPy 已安装')
except ImportError:
    print('✗ NumPy 未安装')
"

echo ""
echo "启动Streamlit应用..."
echo "应用将在浏览器中打开，如果未自动打开，请访问:"
echo "  http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止应用"
echo ""

# 启动Streamlit应用
streamlit run app.py
