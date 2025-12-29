# 显存优化（防止推理时爆显存）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# 
uv run python scripts/train.py pi05_mkrobot_lora --exp_name mkrobot_lora_v1
