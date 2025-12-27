#!/bin/bash
# 获取脚本所在目录 (mkygogo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录 (openpi)，即 mkygogo 的上一级
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project Root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 配置参数
# 1. Checkpoint 路径
CHECKPOINT_DIR="/home/jr/PI/checkpoints/mkrobot_pi05_lora_pickandplace_29999"
# 2. 训练配置名称
CONFIG_NAME="pi05_mkrobot_lora"
# 3. 端口
PORT=8000

echo "Starting Policy Server with uv..."
echo "Config: $CONFIG_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"

# 修正点：
# 1. 加入 policy:checkpoint 子命令
# 2. 移除 --host 参数（脚本不支持，默认监听 0.0.0.0）
uv run python scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$CONFIG_NAME" \
    --policy.dir "$CHECKPOINT_DIR"