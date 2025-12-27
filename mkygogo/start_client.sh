#!/bin/bash
# 获取脚本所在目录 (mkygogo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录 (openpi)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project Root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 自动赋予串口权限 (防止 Permission denied)
if [ -e /dev/ttyACM0 ]; then
    echo "Setting permissions for /dev/ttyACM0..."
    sudo chmod 666 /dev/ttyACM0
fi

# 配置参数
HOST="localhost"   # 如果服务端在另一台机器，请改为服务器 IP
PORT=8000
PROMPT="pick up the small cube and place it in the box" # 提示词
HZ=30 # 控制频率

echo "Starting MKRobot Client..."
echo "Connecting to $HOST:$PORT"

#强制本地连接不走代理
export no_proxy="localhost,127.0.0.1,0.0.0.0"

uv run python mkygogo/mkrobot/main.py \
    --host "$HOST" \
    --port "$PORT" \
    --prompt "$PROMPT" \
    --control_hz "$HZ" \
    --action_horizon 6