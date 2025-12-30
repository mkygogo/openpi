import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import sys

# === 引入 OpenPI 客户端 ===
try:
    from openpi_client import websocket_client_policy
except ImportError:
    print("错误: 找不到 openpi_client 模块。请确保在正确的 python 环境(如 uv run)下运行。")
    sys.exit(1)

# ================= 配置区域 =================
DATASET_ROOT = "/home/jr/PI/data/mkrobot_cube_dataset_backup_56" 
EPISODE_ID = 5
CHUNK_ID = 0 
SERVER_HOST = "localhost"
SERVER_PORT = 8000
PROMPT = "pick up the small cube and place it in the box"
TARGET_SIZE = (224, 224) 
# ===========================================

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
    logger = logging.getLogger("ReplayClient")

    logger.info(f"正在连接 Policy Server ({SERVER_HOST}:{SERVER_PORT})...")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=SERVER_HOST,
            port=SERVER_PORT
        )
        meta = client.get_server_metadata()
        logger.info(f"连接成功! Server Meta: {meta}")
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return

    logger.info(f"正在加载 Episode {EPISODE_ID} 数据...")
    df = load_episode_data(DATASET_ROOT, CHUNK_ID, EPISODE_ID)
    cap_top = get_video_reader(DATASET_ROOT, "top", CHUNK_ID, EPISODE_ID)
    cap_wrist = get_video_reader(DATASET_ROOT, "wrist", CHUNK_ID, EPISODE_ID)
    
    total_frames = len(df)
    logger.info(f"数据加载完毕，共 {total_frames} 帧。开始回放推理...")

    errors = []
    ground_truth_actions = []
    predicted_actions = []

    # 预先创建全黑图像（用于填充缺失的摄像头）
    dummy_img = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)

    for i in range(total_frames):
        row = df.iloc[i]
        
        gt_action = row['action']
        state = row['observation.state']
        
        ret_t, frame_top = cap_top.read()
        ret_w, frame_wrist = cap_wrist.read()
        
        if not ret_t or not ret_w:
            break

        # 1. 图像处理 (BGR -> RGB)
        img_top_rgb = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
        img_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)

        # 2. 缩放
        img_top_resized = cv2.resize(img_top_rgb, TARGET_SIZE)
        img_wrist_resized = cv2.resize(img_wrist_rgb, TARGET_SIZE)

        # 3. 构造请求
        # 这里的键名必须与 Server 报错信息里的 expected keys 完全一致
        request_data = {
            "image": {
                "base_0_rgb": img_top_resized,          # 映射 Top -> Base
                "right_wrist_0_rgb": img_wrist_resized, # 映射 Wrist -> Right Wrist
                "left_wrist_0_rgb": dummy_img           # 填充 Missing -> Dummy Black
            },
            "image_mask": {
                # 真实存在的摄像头，Mask 为 True
                "base_0_rgb": np.array(True, dtype=bool),
                "right_wrist_0_rgb": np.array(True, dtype=bool),
                
                # 填充的假摄像头，Mask 为 False (告诉模型不要看这张图)
                "left_wrist_0_rgb": np.array(False, dtype=bool)
            },
            "state": np.array(state, dtype=np.float32),
            "prompt": PROMPT
        }

        try:
            result = client.infer(request_data)
        except Exception as e:
            logger.error(f"推理请求失败 (Frame {i}): {e}")
            break

        # 4. 提取动作
        action_chunk = result['actions']
        if not isinstance(action_chunk, np.ndarray):
             action_chunk = np.array(action_chunk)
        
        # Pi0 输出的维度可能是 (Batch, Horizon, Dim) -> (1, 50, 7)
        # 我们取第一个 batch，第一个 time step
        pred_action = action_chunk[0] 

        # 5. 计算误差
        mse = np.mean((pred_action - gt_action) ** 2)
        errors.append(mse)
        
        ground_truth_actions.append(gt_action)
        predicted_actions.append(pred_action)

        if i % 20 == 0:
            logger.info(f"Frame {i:03d}/{total_frames} | MSE: {mse:.6f}")

    cap_top.release()
    cap_wrist.release()

    if len(ground_truth_actions) == 0:
        logger.error("未收集到数据，无法绘图。")
        return

    logger.info("生成对比图...")
    gt_array = np.array(ground_truth_actions)
    pred_array = np.array(predicted_actions)
    
    plt.figure(figsize=(12, 6))
    plt.plot(gt_array[:, 0], label='Ground Truth (J1)', color='black', linewidth=2)
    plt.plot(pred_array[:, 0], label='Prediction (J1)', color='red', linestyle='--', alpha=0.8)
    plt.title(f'Real Replay Validation - Joint 1 (Episode {EPISODE_ID})')
    plt.xlabel('Frame')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True)
    
    output_img = 'real_replay_check_j1.png'
    plt.savefig(output_img)
    logger.info(f"图片已保存: {output_img}")
    logger.info(f"全流程平均 MSE: {np.mean(errors):.6f}")

if __name__ == "__main__":
    main()