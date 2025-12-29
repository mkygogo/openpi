import os
import json
import pandas as pd
from pathlib import Path

# === 配置 ===
DATASET_ROOT = "/home/jr/PI/data/mkrobot_cube_dataset_backup_56"  # 你的数据集目录名
OUTPUT_FILE = "dataset_analysis.txt"       # 输出结果文件名

def analyze_dataset(dataset_path, output_file):
    dataset_path = Path(dataset_path)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        def log(msg):
            print(msg)
            f.write(msg + '\n')
            
        log(f"=== Dataset Analysis Report ===")
        log(f"Target Directory: {dataset_path}\n")
        
        if not dataset_path.exists():
            log(f"Error: Directory {dataset_path} not found.")
            return

        # 1. 分析 info.json (基本信息)
        # ------------------------------------------------
        info_path = dataset_path / "meta" / "info.json"
        if info_path.exists():
            log(f"--- [1] Content of meta/info.json ---")
            try:
                with open(info_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    log(json.dumps(data, indent=2, ensure_ascii=False))
            except Exception as e:
                log(f"Error reading info.json: {e}")
        else:
            log("Warning: meta/info.json not found.")
        log("\n")

        # 2. 分析 tasks.jsonl (任务描述)
        # ------------------------------------------------
        tasks_path = dataset_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            log(f"--- [2] Sample tasks from meta/tasks.jsonl (First 2) ---")
            try:
                with open(tasks_path, 'r', encoding='utf-8') as jf:
                    for i, line in enumerate(jf):
                        if i >= 2: break
                        log(f"Task {i}: {line.strip()}")
            except Exception as e:
                log(f"Error reading tasks.jsonl: {e}")
        else:
            log("Warning: meta/tasks.jsonl not found.")
        log("\n")

        # 3. 分析 stats.json (统计信息与维度)
        # ------------------------------------------------
        stats_path = dataset_path / "meta" / "stats.json"
        if stats_path.exists():
            log(f"--- [3] Structure of meta/stats.json ---")
            try:
                with open(stats_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    # 只打印键名和简单的类型/长度，避免打印海量数组
                    for key, value in data.items():
                        if isinstance(value, dict):
                            log(f"Key: '{key}'")
                            for sub_k, sub_v in value.items():
                                if isinstance(sub_v, list):
                                    log(f"  - {sub_k}: List (len={len(sub_v)})")
                                    # 打印前3个数值作为示例，判断归一化范围
                                    if len(sub_v) > 0 and isinstance(sub_v[0], (int, float)):
                                        log(f"    Sample: {sub_v[:3]}...")
                                else:
                                    log(f"  - {sub_k}: {sub_v}")
                        else:
                            log(f"Key: '{key}' -> {type(value)}")
            except Exception as e:
                log(f"Error reading stats.json: {e}")
        log("\n")

        # 4. 分析 Parquet 数据 (实际动作与观测)
        # ------------------------------------------------
        data_dir = dataset_path / "data" / "chunk-000"
        if data_dir.exists():
            parquet_files = sorted(list(data_dir.glob("*.parquet")))
            if parquet_files:
                sample_file = parquet_files[0]
                log(f"--- [4] Analyzing Sample Episode: {sample_file.name} ---")
                try:
                    df = pd.read_parquet(sample_file)
                    log(f"DataFrame Shape (Rows, Cols): {df.shape}")
                    log(f"Columns List: {list(df.columns)}")
                    
                    log("\n--- First Row Data Sample ---")
                    # 转为dict方便查看
                    first_row = df.iloc[0].to_dict()
                    for k, v in first_row.items():
                        # 如果是图像或大数组，只打印类型和形状
                        val_str = str(v)
                        if hasattr(v, 'shape'):
                            val_str = f"Array shape: {v.shape}"
                        elif isinstance(v, list) and len(v) > 10:
                            val_str = f"List len={len(v)} (Sample: {v[:5]}...)"
                        elif isinstance(v, bytes):
                            val_str = f"Bytes (len={len(v)})"
                        
                        log(f"  {k}: {val_str}")
                except Exception as e:
                    log(f"Error reading parquet file: {e}")
            else:
                log("No .parquet files found in data/chunk-000")
        else:
            log(f"Directory {data_dir} not found.")

    print(f"\n[完成] 分析结果已保存到: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    analyze_dataset(DATASET_ROOT, OUTPUT_FILE)