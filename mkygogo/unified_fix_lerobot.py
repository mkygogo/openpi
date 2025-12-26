import pandas as pd
import json
import pathlib
import subprocess
import shutil
import cv2
import os
from tqdm import tqdm

# ================= 配置区 =================
# 1. 你的数据集根目录
DATASET_PATH = pathlib.Path("/home/jr/PI/data/mkrobot_cube_dataset")
# 2. 外部备份目录（必须在 DATASET_PATH 之外，防止被 LeRobot 递归扫描）
EXTERNAL_BACKUP_PATH = pathlib.Path("/home/jr/PI/data_backups/mkrobot_raw_data")
# 3. 任务指令
CORRECT_TASK = "pick up the small cube and place it in the box"
FPS = 30
# =========================================

def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def main():
    print(f"🚀 开始标准化数据集（含深度物理清理）: {DATASET_PATH}")
    
    data_dir = DATASET_PATH / "data" / "chunk-000"
    meta_dir = DATASET_PATH / "meta"
    video_root = DATASET_PATH / "videos"
    
    # --- STEP 0: 物理隔离与深度清理 ---
    print("Step 0: 正在搬离原始文件并清理冗余元数据...")
    EXTERNAL_BACKUP_PATH.mkdir(parents=True, exist_ok=True)

    # 1. 搬离原始 Parquet 聚合文件到外部（如果还在 data 目录下）
    for f in data_dir.glob("file-*.parquet"):
        print(f"📦 搬运原始数据: {f.name}")
        shutil.move(str(f), str(EXTERNAL_BACKUP_PATH / f.name))
        
    # 2. 搬离之前的 raw_backup 目录到外部（防止递归扫描导致数据翻倍）
    old_backup = DATASET_PATH / "data" / "raw_backup"
    if old_backup.exists():
        staging_backup = EXTERNAL_BACKUP_PATH / "raw_backup"
        if not staging_backup.exists():
            print(f"📦 搬运备份目录: {old_backup.name}")
            shutil.move(str(old_backup), str(staging_backup))
        else:
            print(f"🗑️ 删除重复备份目录: {old_backup}")
            shutil.rmtree(old_backup)

    # 3. 删除导致干扰的旧元数据文件
    files_to_delete = [
        meta_dir / "episodes",       # 这是一个文件夹
        meta_dir / "tasks.parquet",  # 冗余文件
        meta_dir / "stats.json"      # 冗余文件
    ]
    for path in files_to_delete:
        if path.exists():
            if path.is_dir():
                print(f"🗑️ 删除目录: {path.name}")
                shutil.rmtree(path)
            else:
                print(f"🗑️ 删除文件: {path.name}")
                path.unlink()

    # --- STEP 1: 重新加载并标准化数据 ---
    print("\nStep 1: 拆分并标准化 Parquet 数据 (Index & Timestamp)...")
    # 从外部备份读取原始数据
    raw_files = sorted(EXTERNAL_BACKUP_PATH.glob("file-*.parquet"))
    if not raw_files:
        print(f"❌ 错误：在 {EXTERNAL_BACKUP_PATH} 下没找到原始文件！")
        return
        
    full_df = pd.concat([pd.read_parquet(f) for f in raw_files], ignore_index=True)
    
    # 清理 data 目录下旧的 episode 文件
    for f in data_dir.glob("episode_*.parquet"): f.unlink()

    episodes = sorted(full_df["episode_index"].unique())
    time_step = 1.0 / FPS
    ep_info_list = []
    
    for ep_idx in tqdm(episodes, desc="处理 Parquet"):
        ep_df = full_df[full_df["episode_index"] == ep_idx].copy().sort_values("index")
        orig_start_idx = ep_df["index"].min()
        num_frames = len(ep_df)
        
        ep_df["index"] = range(num_frames)
        ep_df["timestamp"] = [float(i * time_step) for i in range(num_frames)]
        
        out_path = data_dir / f"episode_{int(ep_idx):06d}.parquet"
        ep_df.to_parquet(out_path, index=False)
        ep_info_list.append({"index": int(ep_idx), "length": num_frames, "orig_start": orig_start_idx})

    # --- STEP 2: 物理裁剪视频并重置 PTS ---
    print("\nStep 2: 帧准确视频裁剪 (重置 PTS 时间戳)...")
    for cam in ["observation.images.top", "observation.images.wrist"]:
        cam_dir = video_root / cam / "chunk-000"
        
        # 查找原始视频（可能在 cam_dir 也可以在外部备份）
        raw_videos = sorted(cam_dir.glob("file-*.mp4"))
        
        video_map = []
        offset = 0
        for v in raw_videos:
            cnt = get_video_frame_count(v)
            video_map.append({"path": v, "start": offset, "end": offset + cnt - 1})
            offset += cnt

        for ep in tqdm(ep_info_list, desc=f"裁剪 {cam}"):
            source = next((v for v in video_map if v["start"] <= ep["orig_start"] <= v["end"]), None)
            if not source: continue
            
            local_start = ep["orig_start"] - source["start"]
            local_end = local_start + ep["length"] - 1
            out_video = cam_dir / f"episode_{ep['index']:06d}.mp4"
            
            cmd = [
                'ffmpeg', '-y', '-i', str(source["path"]),
                '-vf', f"select='between(n,{local_start},{local_end})',setpts=PTS-STARTPTS",
                '-vsync', '0', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                str(out_video)
            ]
            run_cmd(cmd)
            
        # 裁剪完成后，将原始视频大文件搬离，防止干扰索引
        for v in raw_videos:
            dest = EXTERNAL_BACKUP_PATH / v.name
            if not dest.exists():
                print(f"📦 搬离视频原件: {v.name}")
                shutil.move(str(v), str(dest))
            else:
                v.unlink()

    # --- STEP 3: 补全元数据 ---
    print("\nStep 3: 强制刷新元数据 (补全 stats/length/task)...")
    
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": CORRECT_TASK}) + "\n")
    
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in ep_info_list:
            f.write(json.dumps({"episode_index": ep["index"], "tasks": [CORRECT_TASK], "length": ep["length"]}) + "\n")
            
    with open(meta_dir / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for ep in ep_info_list:
            f.write(json.dumps({"episode_index": ep["index"], "stats": {}}) + "\n")

    info_path = meta_dir / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    info.update({
        "codebase_version": "v2.1",
        "total_episodes": len(episodes),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    })
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # --- STEP 4: 清理缓存 ---
    cache_dir = pathlib.Path.home() / ".cache/huggingface/datasets"
    if cache_dir.exists():
        print(f"\n🗑️ 清理数据集缓存: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    print("\n✨ 恭喜！一键标准化与深度物理清理已完成。")
    print(f"原始文件已安全搬运至: {EXTERNAL_BACKUP_PATH}")

if __name__ == "__main__":
    main()