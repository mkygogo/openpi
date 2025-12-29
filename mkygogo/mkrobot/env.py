import numpy as np
import logging
import cv2
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

    def _process_image(self, img_np, target_size=448):
        """
        复刻训练时的图像处理逻辑：中心裁剪 + 缩放
        保持输入输出均为 Numpy [H, W, C] (OpenCV格式)，避免 Client 端引入 Torch 复杂性
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
        pass

    def is_episode_complete(self) -> bool:
        return False

    def get_observation(self) -> Dict:
        raw_obs = self.controller.get_observation()
        
        raw_img_base = raw_obs["images"].get("top")
        raw_img_wrist = raw_obs["images"].get("wrist")
        state = raw_obs["state"]

        if raw_img_base is None: raw_img_base = np.zeros((480, 640, 3), dtype=np.uint8)
        if raw_img_wrist is None: raw_img_wrist = np.zeros((360, 640, 3), dtype=np.uint8)

        # 这里处理后，图像尺寸变为 448x448，且内容经过了中心裁剪
        img_base_processed = self._process_image(raw_img_base, target_size=448)
        img_wrist_processed = self._process_image(raw_img_wrist, target_size=448)

        # ==========================================================
        # 🛠️ DEBUG: 渲染送给模型的图像 (这就真的是模型看到的画面)
        # ==========================================================
        try:
            # OpenCV 的 imshow 默认需要 BGR 格式，但我们的 img_base 是 RGB
            # 如果直接 show，红色物体会变蓝。为了方便人眼观察，我们转回 BGR 显示。
            # (这不影响送给模型的数据，只影响显示的窗口)
            show_base = cv2.cvtColor(img_base_processed, cv2.COLOR_RGB2BGR)
            show_wrist = cv2.cvtColor(img_wrist_processed, cv2.COLOR_RGB2BGR)

            # 在图片上打印当前的 Step，方便截图分析
            cv2.putText(show_base, f"TOP Step: {self.step_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示窗口
            cv2.imshow("Debug View: TOP (Processed)", show_base)
            cv2.imshow("Debug View: WRIST (Processed)", show_wrist)
            
            # 必须调用 waitKey 才能刷新窗口，1ms 延迟
            cv2.waitKey(1)
        except Exception as e:
            print(f"Display Error: {e}")
        # ==========================================================


        # 保存状态供 debug 使用
        self.current_state = state

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
                curr = self.current_state
                act = raw_action
                diff = act - curr
                with np.printoptions(precision=3, suppress=True, linewidth=200):
                    # 重点关注 Act 的最后一位：如果是 1.0 (或接近最大值) 代表闭合，0.0 代表张开
                    logger.info(f"Step {self.step_count}")
                    logger.info(f"  Act  (Model): {act}")  
                    logger.info(f"  Curr (Robot): {curr}")
                
                # 如果差值非常大 (例如 > 0.5 弧度)，说明模型输出的和当前位置完全不匹配
                if np.max(np.abs(diff)) > 0.5:
                    logger.warning("🚨 动作偏差过大！可能是坐标系不匹配或模型未收敛。")

            # 发送给 Controller 执行
            self.controller.apply_action(np.array(raw_action))

    def close(self):
        cv2.destroyAllWindows()
        self.controller.close()