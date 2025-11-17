#!/bin/bash
#
# run.sh: 自动安装、设置并运行 PCBTagent
#
# 此脚本旨在为论文审稿人提供一键式复现体验。
# 它将：
# 1. 创建一个独立的 Conda 环境
# 2. 安装所有 Python 依赖
# 3. 运行 src/main.py 脚本来处理示例数据
#
# 假设:
# - 已安装 Conda (Anaconda / Miniconda)。
# - '.env' 文件已由作者预先配置并包含有效的 API 密钥。
# - 'requirements.txt' 文件存在。
#

# --- 1. 配置 ---
# 定义环境名称
ENV_NAME="pcbt_env"
# 定义 Python 版本 (建议使用一个稳定、常见的版本)
PYTHON_VERSION="3.10"

echo "[INFO] 启动 PCBTagent 自动复现脚本..."

# --- 2. 创建 Conda 环境 ---
echo "[1/3] 正在创建 Conda 环境 '$ENV_NAME' (使用 Python $PYTHON_VERSION)..."
# '-y' 标志自动确认所有提示
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 检查环境是否创建成功
if [ $? -ne 0 ]; then
    echo "[ERROR] Conda 环境创建失败。请检查您的 Conda 安装。"
    exit 1
fi

# --- 3. 安装依赖 ---
echo "[2/3] 正在激活环境并安装依赖 (来自 requirements.txt)..."
# 'conda run' 是在脚本中向指定环境执行命令的推荐方式
conda run -n $ENV_NAME pip install -r requirements.txt

# 检查依赖是否安装成功
if [ $? -ne 0 ]; then
    echo "[ERROR] 依赖安装失败。请检查 'requirements.txt' 文件。"
    exit 1
fi

# --- 4. 运行主脚本 ---
echo "[3/3] 正在运行主脚本 (src/main.py)..."
# 此命令在 $ENV_NAME 环境中执行 main.py
# 脚本将自动从 config.py 和 .env 加载配置
conda run -n $ENV_NAME python src/main.py

# 检查 Python 脚本是否执行成功
if [ $? -ne 0 ]; then
    echo "[ERROR] Python 脚本执行失败。请检查 'logs.log' 文件获取详情。"
    exit 1
fi

echo "=== [成功] 脚本执行完毕。 ==="
echo "输出文件位于您在 config.py 或 .env 中指定的 OUTPUT_PATH。"
echo "若要再次手动运行，请使用: 'conda activate $ENV_NAME' 然后 'python src/main.py'"