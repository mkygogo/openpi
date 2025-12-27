import numpy as np
import logging
from typing import Dict
from openpi_client.runtime import environment

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

    def reset(self) -> None:
        logger.info("Resetting environment...")
        pass

    def is_episode_complete(self) -> bool:
        return False

    def get_observation(self) -> Dict:
        raw_obs = self.controller.get_observation()
        
        img_base = raw_obs["images"].get("top")
        img_wrist = raw_obs["images"].get("wrist")
        state = raw_obs["state"]

        if img_base is None: img_base = np.zeros((480, 640, 3), dtype=np.uint8)
        if img_wrist is None: img_wrist = np.zeros((480, 640, 3), dtype=np.uint8)

        # 保存状态供 debug 使用
        self.current_state = state

        return {
            "image": {
                "base_0_rgb": img_base,
                "left_wrist_0_rgb": img_wrist,
                "right_wrist_0_rgb": img_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.array(True),
                "left_wrist_0_rgb": np.array(True),
                "right_wrist_0_rgb": np.array(True),
            },
            "state": state,
            "prompt": self.prompt 
        }

    def apply_action(self, action: Dict) -> None:
        raw_action = action.get("actions")
        if raw_action is not None:
            if hasattr(raw_action, 'cpu'): raw_action = raw_action.cpu().numpy()
            if hasattr(raw_action, 'numpy'): raw_action = raw_action.numpy()
            
            # --- 🛡️ 调试日志: 状态 vs 动作 ---
            # 每 10 步打印一次，避免刷屏太快
            self.step_count += 1
            if self.step_count % 10 == 0:
                # 打印前 3 个关节的角度对比
                curr = self.current_state[:3]
                act = raw_action[:3]
                diff = act - curr
                logger.info(f"Step {self.step_count} | Curr: {np.round(curr,2)} | Act: {np.round(act,2)} | Diff: {np.round(diff,3)}")
                
                # 如果差值非常大 (例如 > 0.5 弧度)，说明模型输出的和当前位置完全不匹配
                if np.max(np.abs(diff)) > 0.5:
                    logger.warning("🚨 动作偏差过大！可能是坐标系不匹配或模型未收敛。")

            # 发送给 Controller 执行
            self.controller.apply_action(np.array(raw_action))

    def close(self):
        self.controller.close()