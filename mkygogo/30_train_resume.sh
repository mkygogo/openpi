# 1. 显存优化 (保持之前的设置)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# 2. 恢复训练命令
# 变化：去掉了 --overwrite，加上了 --resume
uv run python examples/train.py pi05_mkrobot_lora \
    --exp_name mkrobot_lora_v1 \
    --resume