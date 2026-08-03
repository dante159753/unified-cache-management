#!/bin/bash
set -e

# ============================================
# 配置参数
# ============================================
INSTALL_MODE=${INSTALL_MODE:-"source"}
PIP_INDEX_URL=${PIP_INDEX_URL:-"https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"}
WORKSPACE=${WORKSPACE:-"./"}
# 规范为绝对路径，避免后续 cd 到 SRC_DIR 后 --outdir 的相对路径指向错误位置
WORKSPACE="$(cd "${WORKSPACE}" && pwd)"
SRC_DIR="${WORKSPACE}/unified-cache-management"
OUT_DIR="${WORKSPACE}"
# ============================================
# 设置 pip 源
# ============================================
pip config set global.index-url "${PIP_INDEX_URL}"


export PLATFORM=ascend

# ============================================
# 源码编译模式
# ============================================
if [ "${INSTALL_MODE}" != "package" ]; then
    echo "=== [Mode: source] 安装编译依赖 ==="
    apt-get update
    apt-get install -y --no-install-recommends git
    rm -rf /var/lib/apt/lists/*
    
    pip install --no-cache-dir build cmake
    
    echo "=== [Mode: source] 执行编译脚本 ==="
    export WORKSPACE="${WORKSPACE}"
    export SKIP_TAR=1
    export ENABLE_SPARSE=false
    bash "${SRC_DIR}/scripts/build_ascend.sh"
else
    echo "=== [Mode: package] 使用预置源码 ==="
fi

# ============================================
# 统一构建 whl 包
# ============================================
echo "=== 构建 ucm whl 包 ==="
mkdir -p "${OUT_DIR}"
cd "${SRC_DIR}"

# 确保 build 工具已安装
pip install --no-cache-dir build

# 构建 wheel
python -m build --wheel --outdir "${OUT_DIR}"

echo "=== 构建完成，输出文件 ==="
ls -la "${OUT_DIR}"/*.whl

echo "=== whl 包信息 ==="
WHL_FILE="$(ls "${OUT_DIR}"/*.whl 2>/dev/null | head -1)"
if [ -n "${WHL_FILE}" ]; then
    echo "whl 文件已生成: ${WHL_FILE}"
else
    echo "错误: 未找到 whl 文件" >&2
    exit 1
fi