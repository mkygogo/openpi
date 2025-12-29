import pandas as pd
import json
import pathlib
import cv2
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
# 必须与转换脚本中的路径一致
DATASET_PATH = pathlib.Path("/home/jr/PI/data/mkrobot_cube_dataset")
FPS = 30
# 允许的帧数误差（FFmpeg有时候会多/少算一帧，通常允许1帧误差）
FRAME_TOLERANCE = 1 
# =========================================

class ValidationReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.total_episodes = 0
        self.valid_episodes = 0

    def error(self, ep_idx, msg):
        self.errors.append(f"[Episode {ep_idx:06d}] ❌ {msg}")

    def warning(self, ep_idx, msg):
        self.warnings.append(f"[Episode {ep_idx:06d}] ⚠️ {msg}")

def get_video_info(video_path):
    if not video_path.exists():
        return False, 0, "File not found"
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, 0, "Cannot open video"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return True, frame_count, {"fps": fps, "res": (width, height)}
    except Exception as e:
        return False, 0, str(e)

def validate():
    print(f"🔍 开始检查数据集: {DATASET_PATH}")
    report = ValidationReport()
    
    # 1. 检查 Meta 目录结构
    meta_dir = DATASET_PATH / "meta"
    data_dir = DATASET_PATH / "data" / "chunk-000"
    video_root = DATASET_PATH / "videos"
    
    required_meta = ["info.json", "episodes.jsonl", "tasks.jsonl"]
    for f in required_meta:
        if not (meta_dir / f).exists():
            print(f"⛔ 严重错误: 缺少元数据文件 {f}")
            return

    # 2. 读取 Episodes 列表
    episodes_meta = []
    try:
        with open(meta_dir / "episodes.jsonl", "r") as f:
            for line in f:
                episodes_meta.append(json.loads(line))
    except Exception as e:
        print(f"⛔ 读取 episodes.jsonl 失败: {e}")
        return

    report.total_episodes = len(episodes_meta)
    print(f"📋 发现 {report.total_episodes} 个 Episode 元数据记录，开始逐个校验...")

    # 3. 逐个 Episode 校验
    for ep_info in tqdm(episodes_meta, desc="校验中"):
        ep_idx = ep_info["episode_index"]
        is_valid = True
        
        # --- A. 检查 Parquet ---
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        parquet_rows = 0
        
        if not parquet_path.exists():
            report.error(ep_idx, f"缺少 Parquet 文件: {parquet_path.name}")
            is_valid = False
        else:
            try:
                df = pd.read_parquet(parquet_path)
                parquet_rows = len(df)
                
                # 检查 Index 连续性
                if not df["index"].is_monotonic_increasing:
                    report.error(ep_idx, "Parquet 'index' 列不是单调递增的")
                    is_valid = False
                if df["index"].min() != 0:
                    report.error(ep_idx, f"Parquet 'index' 不从 0 开始 (Start: {df['index'].min()})")
                    is_valid = False
                
                # 检查 Timestamp 逻辑
                expected_duration = (parquet_rows - 1) / FPS
                last_ts = df["timestamp"].iloc[-1]
                if abs(last_ts - expected_duration) > 0.1: # 允许0.1秒误差
                    report.warning(ep_idx, f"时间戳可能未重置? Last TS: {last_ts:.2f}, Expected: {expected_duration:.2f}")

            except Exception as e:
                report.error(ep_idx, f"Parquet 读取损坏: {e}")
                is_valid = False

        # --- B. 检查 Videos (Top & Wrist) ---
        # 假设 info.json 里没写具体的 key，我们默认检查转换脚本里用到的 camera names
        cameras = ["observation.images.top", "observation.images.wrist"]
        
        for cam in cameras:
            video_path = video_root / cam / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
            exists, v_frames, v_info = get_video_info(video_path)
            
            if not exists:
                report.error(ep_idx, f"缺少视频 ({cam}): {video_path.name}")
                is_valid = False
                continue
            
            # 检查 FPS 匹配
            if abs(v_info['fps'] - FPS) > 1.0:
                report.warning(ep_idx, f"视频 FPS ({v_info['fps']}) 与设定 ({FPS}) 不符")
            
            # --- C. 核心检查: 帧数同步 ---
            # 只有当 Parquet 也读取成功时才对比
            if parquet_rows > 0:
                diff = abs(parquet_rows - v_frames)
                if diff > FRAME_TOLERANCE:
                    report.error(ep_idx, f"严重失步! {cam} -> Parquet行数: {parquet_rows}, 视频帧数: {v_frames} (Diff: {diff})")
                    is_valid = False
                elif diff > 0:
                    # 警告但不标记为 Invalid (FFmpeg 常见误差)
                    report.warning(ep_idx, f"轻微帧数差异 {cam} -> Parquet: {parquet_rows}, Video: {v_frames}")

        if is_valid:
            report.valid_episodes += 1

    # ================= 打印报告 =================
    print("\n" + "="*40)
    print("📢 校验报告 SUMMARY")
    print("="*40)
    print(f"总集数: {report.total_episodes}")
    print(f"✅ 合格: {report.valid_episodes}")
    print(f"❌ 失败: {len(report.errors) > 0}")
    
    if report.warnings:
        print(f"\n⚠️ 警告 ({len(report.warnings)}):")
        # 只打印前10个警告，避免刷屏
        for w in report.warnings[:10]:
            print(w)
        if len(report.warnings) > 10: print(f"... 以及其他 {len(report.warnings)-10} 个警告")

    if report.errors:
        print(f"\n❌ 错误 ({len(report.errors)}):")
        for e in report.errors:
            print(e)
        print("\n结论: 建议修复上述错误后再进行训练。")
    else:
        print("\n🎉 完美！数据集结构完整，音画同步 (Parquet/Video Aligned)。")
        print("可以直接用于 OpenPi / LeRobot 训练。")

if __name__ == "__main__":
    validate()