import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import sys

try:
    from openpi_client import websocket_client_policy
except ImportError:
    print("错误: 找不到 openpi_client 模块。")
    sys.exit(1)

# ================= 配置区域 =================
DATASET_ROOT = "/home/jr/PI/data/mkrobot_cube_dataset_backup_56" 
EPISODE_ID = 6
CHUNK_ID = 0 
SERVER_HOST = "localhost"
SERVER_PORT = 8000
PROMPT = "pick up the small cube and place it in the box"
TARGET_SIZE = (224, 224) 
# ===========================================

# 关节名称映射，方便看图
JOINT_NAMES = [
    "Joint 1 (Base)", "Joint 2 (Shoulder)", "Joint 3 (Elbow)",
    "Joint 4 (Wrist 1)", "Joint 5 (Wrist 2)", "Joint 6 (Wrist 3)",
    "Joint 7 (Gripper)"
]

def load_episode_data(dataset_root, chunk_id, episode_id):
    parquet_path = os.path.join(
        dataset_root, "data", f"chunk-{chunk_id:03d}", f"episode_{episode_id:06d}.parquet"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"找不到文件: {parquet_path}")
    return pd.read_parquet(parquet_path)

def get_video_reader(dataset_root, camera_name, chunk_id, episode_id):
    video_path = os.path.join(
        dataset_root, "videos", f"observation.images.{camera_name}", 
        f"chunk-{chunk_id:03d}", f"episode_{episode_id:06d}.mp4"
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    return cap

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ReplaySmoothAll")

    # 连接 Server
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
        client.get_server_metadata()
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return

    # 加载数据
    logger.info(f"正在加载 Episode {EPISODE_ID}...")
    df = load_episode_data(DATASET_ROOT, CHUNK_ID, EPISODE_ID)
    cap_top = get_video_reader(DATASET_ROOT, "top", CHUNK_ID, EPISODE_ID)
    cap_wrist = get_video_reader(DATASET_ROOT, "wrist", CHUNK_ID, EPISODE_ID)
    
    total_frames = len(df)
    
    # === 🚀 平滑算法的核心数据结构 ===
    MAX_HORIZON = 100
    action_accumulator = np.zeros((total_frames + MAX_HORIZON, 7), dtype=np.float32)
    count_accumulator = np.zeros((total_frames + MAX_HORIZON, 1), dtype=np.float32)
    
    naive_predictions = []
    ground_truth_actions = []

    logger.info(f"开始推理并执行时间聚合 (共 {total_frames} 帧)...")
    
    dummy_img = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)

    for i in range(total_frames):
        row = df.iloc[i]
        ground_truth_actions.append(row['action'])

        state = row['observation.state'] # <--- 这里读出来的就是真值
        if i == 0:
            state_arr = np.array(state)
            print("\n" + "="*40)
            print(f"🧐 [真相时刻] 数据集里的 State 形状: {state_arr.shape}")
            print(f"🧐 [真相时刻] 数据内容: {state_arr}")
            print("="*40 + "\n")

        ret_t, frame_top = cap_top.read()
        ret_w, frame_wrist = cap_wrist.read()
        if not ret_t or not ret_w: break

        img_top = cv2.resize(cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB), TARGET_SIZE)
        img_wrist = cv2.resize(cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB), TARGET_SIZE)

        request_data = {
            "image": {
                "base_0_rgb": img_top,
                "right_wrist_0_rgb": img_wrist,
                "left_wrist_0_rgb": dummy_img
            },
            "image_mask": {
                "base_0_rgb": np.array(True),
                "right_wrist_0_rgb": np.array(True),
                "left_wrist_0_rgb": np.array(False)
            },
            "state": np.array(row['observation.state'], dtype=np.float32),
            "prompt": PROMPT
        }

        try:
            result = client.infer(request_data)
            action_chunk = np.array(result['actions'])
            if action_chunk.ndim == 3: action_chunk = action_chunk[0]
        except:
            action_chunk = np.zeros((1, 7))

        # 1. 记录朴素预测 (第1帧)
        naive_predictions.append(action_chunk[0])

        # 2. 执行时间聚合
        horizon = action_chunk.shape[0]
        for t in range(horizon):
            if i + t < len(action_accumulator):
                action_accumulator[i + t] += action_chunk[t]
                count_accumulator[i + t] += 1.0

        if i % 50 == 0:
            print(f"Processing frame {i}/{total_frames}...", end='\r')

    cap_top.release()
    cap_wrist.release()

    # 计算最终平滑结果
    count_accumulator[count_accumulator == 0] = 1.0
    smoothed_actions = action_accumulator[:total_frames] / count_accumulator[:total_frames]
    
    naive_array = np.array(naive_predictions)
    gt_array = np.array(ground_truth_actions)

    # === 绘图对比 (所有7个关节) ===
    logger.info("生成全关节对比图...")
    # 设置一个较高的画布，容纳7张图
    fig, axes = plt.subplots(7, 1, figsize=(15, 25), sharex=True)
    
    colors = ['black', 'red', 'blue']
    labels = ['Ground Truth', 'Single Step (Naive)', 'Smoothed (Aggregation)']
    linestyles = ['-', ':', '-']
    alphas = [1.0, 0.7, 1.0]

    for j_idx in range(7):
        ax = axes[j_idx]
        
        # 1. Truth
        ax.plot(gt_array[:, j_idx], color=colors[0], label=labels[0], linewidth=2)
        # 2. Naive
        ax.plot(naive_array[:, j_idx], color=colors[1], linestyle=linestyles[1], label=labels[1], alpha=alphas[1])
        # 3. Smoothed
        ax.plot(smoothed_actions[:, j_idx], color=colors[2], label=labels[2], linewidth=2)
        
        ax.set_title(f'{JOINT_NAMES[j_idx]} Comparison')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        # 只在第一个子图显示图例，节省空间
        if j_idx == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Frame') # 只在最后一张图显示X轴标签
    
    plt.tight_layout()
    save_path = 'smooth_check_all_joints.png'
    plt.savefig(save_path)
    logger.info(f"全关节对比图已保存为: {save_path}")

if __name__ == "__main__":
    main()