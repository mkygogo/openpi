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
        camera_indices = {
                'top':   {'index': 0, 'width': 640, 'height': 480},
                'wrist': {'index': 2, 'width': 640, 'height': 360}
                }

        # 使用 Controller 封装
        self.controller = MKController(port=port, camera_indices=camera_indices)
        self.controller.connect()
        self.step_count = 0
        # 记录上一次执行的动作，用于插值计算
        self.prev_action = None

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
        self.prev_action = None
        return self.get_observation()

    def is_episode_complete(self) -> bool:
        #logger.info("is_episode_complete")
        return False

    def get_observation(self) -> Dict:

        # === ⏱️ [DEBUG] 测速 ===
        # import time
        # now = time.time()
        # if hasattr(self, '_last_loop_time'):
        #     dt = now - self._last_loop_time
        #     fps = 1.0 / dt if dt > 0 else 0
        #     #print(f"⚡ [Env] 实际循环频率: {fps:.1f} Hz (耗时: {dt*1000:.1f} ms)")
        # self._last_loop_time = now
        # =======================
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

    # def _run_interpolation(self, start_pose: np.ndarray, target_pose: np.ndarray, steps: int, dt: float) -> None:
    #     """
    #     [辅助函数] 执行插值动作
    #     - 机械臂 (0-5): 线性插值
    #     - 夹爪 (6): 保持 Start 状态 (不插值)
    #     """
    #     if steps <= 0: return

    #     # === 1. 分离关节和夹爪 ===
    #     start_arm = start_pose[:6]
    #     start_gripper = start_pose[6:]
        
    #     target_arm = target_pose[:6]
    #     # target_gripper = target_pose[6:] # 暂不使用，我们选择在插值期间锁死 Start 状态

    #     for j in range(1, steps + 1):
    #         alpha = j / (steps + 1)
            
    #         # === 2. 机械臂插值 (Linear) ===
    #         interp_arm = start_arm + (target_arm - start_arm) * alpha
            
    #         # === 3. 夹爪不插值 ===
    #         # 策略：在赶路期间，保持上一帧的夹爪状态，防止半开半闭
    #         # 等赶路结束（进入主循环），夹爪会瞬间变成 new_chunk 的第0帧状态
    #         interp_gripper = start_gripper 
            
    #         # 组合
    #         interp_cmd = np.concatenate([interp_arm, interp_gripper])
            
    #         self.controller.apply_action(interp_cmd)
    #         time.sleep(dt)

    def _run_interpolation(self, start_pose: np.ndarray, target_pose: np.ndarray, steps: int, dt: float) -> None:
        """
        [辅助函数] 执行插值动作 (已优化控频)
        """
        if steps <= 0: return

        start_arm = start_pose[:6]
        start_gripper = start_pose[6:]
        target_arm = target_pose[:6]

        for j in range(1, steps + 1):
            # ⏱️ [优化] 记录循环开始时间
            loop_start = time.time()
            
            alpha = j / (steps + 1)
            
            # 机械臂插值
            interp_arm = start_arm + (target_arm - start_arm) * alpha
            # 夹爪保持
            interp_gripper = start_gripper 
            
            interp_cmd = np.concatenate([interp_arm, interp_gripper])
            
            self.controller.apply_action(interp_cmd)
            
            # ⏱️ [优化] 扣除通讯耗时，精确休眠
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def apply_action(self, action: Dict[str, Any]) -> None:
        raw_action = action.get("actions")
        if raw_action is None: return

        if not isinstance(raw_action, np.ndarray):
            raw_action = np.array(raw_action, dtype=np.float32)

        if raw_action.ndim == 1:
            raw_action = raw_action.reshape(1, -1)
        if raw_action.ndim == 3:
            raw_action = raw_action[0]
            
        chunk_len = raw_action.shape[0]
        control_hz = 30.0
        dt = 1.0 / control_hz
        
        # 插值参数
        INTERP_STEPS = 15
        
        # === 🌟 [核心修改] 读取标签 ===
        # 默认为 True 是为了兼容测试脚本，但在实际运行中 Broker 会传 False 过来
        is_new_chunk = action.get("is_new_chunk", True)
        
        # 处理 numpy bool 类型
        if hasattr(is_new_chunk, "item"):
            is_new_chunk = is_new_chunk.item()
            
        # 只有当 (是新Chunk) 且 (不是第一次运行) 时，才进行插值
        should_interpolate = is_new_chunk and (self.prev_action is not None)
        
        # ==========================================
        # 🚀 阶段 1: 处理 Chunk 间的缝隙 (插值)
        # ==========================================
        if should_interpolate:
            print(f"🌊 [Env] 检测到 Chunk 切换，正在执行平滑插值...")
            self._run_interpolation(
                start_pose=self.prev_action, 
                target_pose=raw_action[0], 
                steps=INTERP_STEPS, 
                dt=dt
            )

        # ==========================================
        # 🚀 阶段 2: 原样执行 (全速运行！)
        # ==========================================
        for i in range(chunk_len):
            loop_start = time.time()
            
            final_cmd = raw_action[i]
            
            self.controller.apply_action(final_cmd)   
            
            self.prev_action = final_cmd
            
            if chunk_len > 1:
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # def apply_action(self, action: Dict[str, Any]) -> None:
    #     raw_action = action.get("actions")
    #     if raw_action is None: return

    #     if not isinstance(raw_action, np.ndarray):
    #         raw_action = np.array(raw_action, dtype=np.float32)

    #     # 维度标准化
    #     if raw_action.ndim == 1:
    #         raw_action = raw_action.reshape(1, -1)
    #     if raw_action.ndim == 3:
    #         raw_action = raw_action[0]
            
    #     chunk_len = raw_action.shape[0]
    #     control_hz = 30.0
    #     dt = 1.0 / control_hz
        
    #     INTERP_STEPS = 10
    #     #处理 Chunk 间的缝隙 (插值)
    #     if self.prev_action is not None:
    #         # 调用封装好的函数
    #         self._run_interpolation(
    #             start_pose=self.prev_action, 
    #             target_pose=raw_action[0], 
    #             steps=INTERP_STEPS, 
    #             dt=dt
    #         )
    #     for i in range(chunk_len):
    #         loop_start = time.time()           
    #         final_cmd = raw_action[i]
    #         self.controller.apply_action(final_cmd)   
    #         self.prev_action = final_cmd
            
    #         if chunk_len > 1:
    #             elapsed = time.time() - loop_start
    #             sleep_time = dt - elapsed
    #             if sleep_time > 0:
    #                 time.sleep(sleep_time)

    # def apply_action(self, action: Dict[str, Any]) -> None:
    #     """
    #     [修正版] 分块流式执行 + 对接 Controller 安全层
    #     """
    #     #print(f"🐛 [Main] After squeeze: {raw_action.shape}")
    #     raw_action = action.get("actions")
    #     if raw_action is None: return

    #     # 1. 转换为 Numpy
    #     if not isinstance(raw_action, np.ndarray):
    #         raw_action = np.array(raw_action, dtype=np.float32)

    #     #print(f"🐛 [Env] Raw action shape: {raw_action.shape}, ndim={raw_action.ndim}")

    #     # 2. 维度标准化 (处理 (7,) 或 (1, N, 7))
    #     if raw_action.ndim == 1:
    #         raw_action = raw_action.reshape(1, -1)
    #     if raw_action.ndim == 3:
    #         raw_action = raw_action[0]
    #     # 此时 raw_action 是 (N, 7)，比如 (25, 7)
    #     # 3. 循环执行 Chunk
    #     chunk_len = raw_action.shape[0]
    #     #print(f"🐛 [Env] Chunk execution: len={chunk_len}")
    #     #if chunk_len > 7:
    #     #    print("🐛 [Env] ⚠️ CAUTION: Chunk length > 7, checking loop logic...")

    #     control_hz = 30.0
    #     dt = 1.0 / control_hz
        
    #     for i in range(chunk_len):
    #         loop_start = time.time()
    #         #print(f"🐛 [Env] Loop i={i}/{chunk_len}, accessing raw_action[{i}]")
    #         # 取出单帧 (7,)
    #         single_step = raw_action[i]
            
    #         if single_step.shape != (7,):
    #             print(f"🐛 [Env] ❌ ERROR: Single step shape wrong! {single_step.shape}")

    #         self.controller.apply_action(single_step)   
    #         # 控频
    #         elapsed = time.time() - loop_start
    #         sleep_time = dt - elapsed
    #         if sleep_time > 0:
    #             time.sleep(sleep_time)

    def close(self):
        cv2.destroyAllWindows()
        # 确保 controller 有 close 方法，如果没有会报错
        if hasattr(self.controller, "close"):
            self.controller.close()