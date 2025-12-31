import numpy as np
import logging
import cv2
from typing import Dict, Any
from openpi_client.runtime import environment
import time
from gymnasium import spaces

# 改用 Controller
from mkygogo.mkrobot.mk_controller import MKController

logger = logging.getLogger(__name__)

class MKRobotOpenPIEnv(environment.Environment):
    def __init__(self, prompt: str, port: str = "/dev/ttyACM0"):
        self.prompt = prompt
        camera_indices = {"top": 0, "wrist": 2}
        
        # 使用 Controller 封装
        self.controller = MKController(port=port, camera_indices=camera_indices)
        self.controller.connect()
        self.step_count = 0

    @property
    def action_space(self):
        # 7个维度：6个关节 + 1个夹爪
        # 范围可以写大一点，主要是维度 (7,) 要对
        return spaces.Box(low=-3.14, high=3.14, shape=(7,), dtype=np.float32)

    @property
    def observation_space(self):
        # 同样定义观测空间为 7 维
        return spaces.Dict({
            "state": spaces.Box(low=-3.14, high=3.14, shape=(7,), dtype=np.float32),
            "images": spaces.Dict({
                "top": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            })
        })

    def _process_image(self, img_np, target_size=448):
        """
        复刻训练时的图像处理逻辑：中心裁剪 + 缩放
        """
        if img_np is None:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

        h, w = img_np.shape[:2]
        min_dim = min(h, w)
        
        # 1. 中心裁剪 (Center Crop)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        img_cropped = img_np[start_h:start_h + min_dim, start_w:start_w + min_dim]
        
        # 2. 缩放 (Resize)
        img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return img_resized

    def reset(self) -> None:
        logger.info("Resetting environment...")
        return self.get_observation()

    def is_episode_complete(self) -> bool:
        return False

    def get_observation(self) -> Dict:
        raw_obs = self.controller.get_observation()
        
        # 安全获取
        images = raw_obs.get("images", {})
        raw_img_base = images.get("top")
        raw_img_wrist = images.get("wrist")
        
        state = raw_obs.get("state")
        
        if state is None: 
            state = np.zeros(7, dtype=np.float32)
        
        # 可选：加个安全截断，防止万一 driver 抽风发多了
        if state.shape[0] > 7:
            state = state[:7]

        # 图像容错
        if raw_img_base is None: raw_img_base = np.zeros((480, 640, 3), dtype=np.uint8)
        if raw_img_wrist is None: raw_img_wrist = np.zeros((360, 640, 3), dtype=np.uint8)

        img_base_processed = self._process_image(raw_img_base, target_size=448)
        img_wrist_processed = self._process_image(raw_img_wrist, target_size=448)
        
        # DEBUG View 
        try:
            self.step_count += 1
            show_base = cv2.cvtColor(img_base_processed, cv2.COLOR_RGB2BGR)
            show_wrist = cv2.cvtColor(img_wrist_processed, cv2.COLOR_RGB2BGR)
            cv2.putText(show_base, f"Step: {self.step_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Debug View: TOP", show_base)
            cv2.imshow("Debug View: WRIST", show_wrist)
            cv2.waitKey(1)
        except Exception: pass

        self.current_state = {"state": state}

        return {
            "image": {
                "base_0_rgb": img_base_processed,
                "left_wrist_0_rgb": img_wrist_processed,
                "right_wrist_0_rgb": img_wrist_processed,
            },
            "image_mask": {
                "base_0_rgb": np.array(True),
                "left_wrist_0_rgb": np.array(True),
                "right_wrist_0_rgb": np.array(True),
            },
            "state": state,
            "prompt": self.prompt 
        }

    def apply_action(self, action: Dict[str, Any]) -> None:
        """
        [修正版] 分块流式执行 + 对接 Controller 安全层
        """
        raw_action = action.get("actions")
        if raw_action is None: return

        # 1. 转换为 Numpy
        if not isinstance(raw_action, np.ndarray):
            raw_action = np.array(raw_action, dtype=np.float32)

        # 2. 维度标准化 (处理 (7,) 或 (1, N, 7))
        if raw_action.ndim == 1:
            raw_action = raw_action.reshape(1, -1)
        if raw_action.ndim == 3:
            raw_action = raw_action[0]
        # 此时 raw_action 是 (N, 7)，比如 (25, 7)
        # 3. 循环执行 Chunk
        chunk_len = raw_action.shape[0]
        if chunk_len > 1:
            print(f"\n📦 [Env] Start Chunk: {chunk_len} frames")

        control_hz = 30.0
        dt = 1.0 / control_hz
        
        for i in range(chunk_len):
            loop_start = time.time()
            
            # 取出单帧 (7,)
            single_step = raw_action[i]
            
            if chunk_len > 1:
                if i == 0 or i == chunk_len - 1 or i % 10 == 0:
                    # 只打印第一个关节的值用于观察
                    print(f"   -> Step {i+1:02d}/{chunk_len} | J1: {single_step[0]:.4f} ...")

            self.controller.apply_action(single_step)   
            # 控频
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self):
        cv2.destroyAllWindows()
        # 确保 controller 有 close 方法，如果没有会报错
        if hasattr(self.controller, "close"):
            self.controller.close()