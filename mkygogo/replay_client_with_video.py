import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import sys
from tqdm import tqdm

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
OUTPUT_VIDEO = "replay_render.mp4"
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

def draw_bar(img, x, y, w, h, val, min_v, max_v, color, label):
    """ 在图像上绘制一个数值条 """
    # 背景
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    # 归一化长度
    norm_val = (val - min_v) / (max_v - min_v + 1e-6)
    norm_val = np.clip(norm_val, 0, 1)
    bar_w = int(w * norm_val)
    # 进度
    cv2.rectangle(img, (x, y), (x + bar_w, y + h), color, -1)
    # 边框
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    # 文字
    cv2.putText(img, f"{label}: {val:.3f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ReplayVideo")

    # 1. 连接
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
        client.get_server_metadata()
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return

    # 2. 加载数据
    logger.info(f"加载数据 Episode {EPISODE_ID}...")
    df = load_episode_data(DATASET_ROOT, CHUNK_ID, EPISODE_ID)
    total_frames = len(df)
    
    # 3. 准备平滑计算
    MAX_HORIZON = 100
    action_accumulator = np.zeros((total_frames + MAX_HORIZON, 7), dtype=np.float32)
    count_accumulator = np.zeros((total_frames + MAX_HORIZON, 1), dtype=np.float32)
    
    naive_predictions = []
    ground_truth_actions = []

    cap_top = get_video_reader(DATASET_ROOT, "top", CHUNK_ID, EPISODE_ID)
    cap_wrist = get_video_reader(DATASET_ROOT, "wrist", CHUNK_ID, EPISODE_ID)
    dummy_img = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)

    logger.info(">>> 阶段1/2: 正在推理并计算平滑轨迹...")
    
    for i in tqdm(range(total_frames)):
        row = df.iloc[i]
        ground_truth_actions.append(row['action'])
        
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

        # 记录
        naive_predictions.append(action_chunk[0])

        # 累加 (时间聚合)
        horizon = action_chunk.shape[0]
        for t in range(horizon):
            if i + t < len(action_accumulator):
                action_accumulator[i + t] += action_chunk[t]
                count_accumulator[i + t] += 1.0

    # 释放并重新打开（为了生成视频）
    cap_top.release()
    cap_wrist.release()

    # 4. 计算最终平滑曲线
    count_accumulator[count_accumulator == 0] = 1.0
    smoothed_actions = action_accumulator[:total_frames] / count_accumulator[:total_frames]
    naive_array = np.array(naive_predictions)
    gt_array = np.array(ground_truth_actions)

    # 5. 生成视频
    logger.info(f">>> 阶段2/2: 正在渲染视频 {OUTPUT_VIDEO}...")
    
    # 重新读取
    cap_top = get_video_reader(DATASET_ROOT, "top", CHUNK_ID, EPISODE_ID)
    cap_wrist = get_video_reader(DATASET_ROOT, "wrist", CHUNK_ID, EPISODE_ID)
    
    # 初始化 VideoWriter
    # 画布大小: 左边Top(224) + 中间Wrist(224) + 右边数据板(300) = 748 宽
    # 高度: 224
    canvas_w, canvas_h = 224 + 224 + 320, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30.0, (canvas_w, canvas_h))

    for i in tqdm(range(total_frames)):
        ret_t, frame_top = cap_top.read()
        ret_w, frame_wrist = cap_wrist.read()
        if not ret_t or not ret_w: break
        
        # 画面
        show_top = cv2.resize(frame_top, (224, 224))
        show_wrist = cv2.resize(frame_wrist, (224, 224))
        
        # 数据板
        panel = np.zeros((224, 320, 3), dtype=np.uint8)
        
        # 绘制数值对比 (以 Joint 1 和 Joint 4 为例)
        cv2.putText(panel, f"Frame: {i:03d}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Joint 1
        j1_gt = gt_array[i][0]
        j1_naive = naive_array[i][0]
        j1_smooth = smoothed_actions[i][0]
        
        # 范围假设 (根据你的机器人调整，这里假设 -3 到 3)
        min_v, max_v = -3.0, 3.0 
        
        y_off = 50
        draw_bar(panel, 10, y_off, 280, 15, j1_gt, min_v, max_v, (100, 100, 100), "J1 GT (Gray)")
        draw_bar(panel, 10, y_off+30, 280, 15, j1_naive, min_v, max_v, (0, 0, 255), "J1 Naive (Red)")
        draw_bar(panel, 10, y_off+60, 280, 15, j1_smooth, min_v, max_v, (255, 100, 0), "J1 Smooth (Blue)")
        
        # Joint 4 (Wrist)
        j4_gt = gt_array[i][3]
        j4_naive = naive_array[i][3]
        j4_smooth = smoothed_actions[i][3]
        
        y_off = 150
        draw_bar(panel, 10, y_off, 280, 15, j4_gt, -1.7, 1.2, (100, 100, 100), "J4 GT (Gray)")
        draw_bar(panel, 10, y_off+30, 280, 15, j4_naive, -1.7, 1.2, (0, 0, 255), "J4 Naive (Red)")
        draw_bar(panel, 10, y_off+60, 280, 15, j4_smooth, -1.7, 1.2, (255, 100, 0), "J4 Smooth (Blue)")

        # 拼合
        row = np.hstack([show_top, show_wrist, panel])
        writer.write(row)

    writer.release()
    cap_top.release()
    cap_wrist.release()
    logger.info(f"视频渲染完成! 已保存至 {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()