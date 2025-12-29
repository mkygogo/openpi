import time
import logging
import numpy as np
import serial
import cv2
from typing import Dict, Any

# 尝试导入电机驱动
try:
    from .drivers.DM_Control_Python.DM_CAN import Motor, MotorControl, DM_Motor_Type, Control_Type, DM_variable
    DRIVERS_AVAILABLE = True
except ImportError:
    print("Warning: DM_CAN drivers not found. Hardware will not work.")
    DRIVERS_AVAILABLE = False
    class Motor: pass
    class MotorControl: pass
    class DM_Motor_Type: DM4340=0; DM4310=1

logger = logging.getLogger("MKDriver")

# Sim (URDF) <-> Real (Motor) 方向映射
HARDWARE_DIR = np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

# 关节物理限位 (仅作为最后一道防线，逻辑上由模型保证)
JOINT_LIMITS = {
    0: (-3.0, 3.0), 1: (0.0, 3.0), 2: (0.0, 3.0),
    3: (-1.7, 1.2), 4: (-0.4, 0.4), 5: (-2.0, 2.0),
}

class MKRobotStandalone:
    def __init__(self, port: str = "/dev/ttyACM0", camera_indices: Dict[str, int] = None):
        self.port = port
        self.camera_indices = camera_indices or {}
        self.cameras = {}
        self.serial_conn = None
        self.control = None
        self.is_connected = False
        
        # 参数
        self.max_step_rad = 0.8
        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = -4.7
        
        # 【核心逻辑回归】记录上电时的初始位置
        self.start_joints = np.zeros(6)

        # 初始化电机
        if DRIVERS_AVAILABLE:
            self.motors = {
                "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
                "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
                "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
                "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
                "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
                "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
                "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
            }
        else:
            self.motors = {}

    def connect(self):
        if self.is_connected: return
        
        # 连接机械臂
        if DRIVERS_AVAILABLE:
            try:
                self.serial_conn = serial.Serial(self.port, 921600, timeout=0.5)
                self.control = MotorControl(self.serial_conn)
                self._init_motors()
                self.is_connected = True
                logger.info("Arm Connected.")
            except Exception as e:
                logger.error(f"Failed to connect arm: {e}")
                # 连接失败应抛出异常，不要继续
                raise e
        
        # 连接相机
        for name, idx in self.camera_indices.items():
            try:
                cap = cv2.VideoCapture(idx)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cameras[name] = cap
            except Exception:
                pass

    def _init_motors(self):
        """初始化并设定零点"""
        for name, motor in self.motors.items():
            self.control.addMotor(motor)
            time.sleep(0.01)
            
            # 切换模式并使能
            self.control.switchControlMode(motor, Control_Type.POS_VEL)
            self.control.enable(motor)
            
            # 【重要】不强制修改PID，使用电机内部默认参数（您之前的经验证明它是好用的）
            # self.control.change_motor_param(...) 
            
            logger.info(f"{name} enabled.")

        # 【核心逻辑回归】读取当前物理位置作为零点
        # 确保这和您采集数据时的行为一致
        time.sleep(0.1) # 等一下数据稳定
        self.start_joints = self._read_physical_joints_raw()
        logger.info(f"📍 Set Zero Point at: {self.start_joints}")
        logger.info("机械臂已就绪。现在的姿态被定义为 [0, 0, 0, 0, 0, 0]。")

    def _read_physical_joints_raw(self) -> np.ndarray:
        """读取绝对物理角度 (底层原始值)"""
        pos = []
        for i in range(1, 7):
            m = self.motors[f"joint_{i}"]
            self.control.refresh_motor_status(m)
            pos.append(m.getPosition())
        return np.array(pos)

    def get_observation(self) -> Dict[str, Any]:
        obs = {}
        state = np.zeros(7, dtype=np.float32)

        if self.is_connected:
            try:
                # 1. 读取绝对物理角度
                q_real_abs = self._read_physical_joints_raw()
                
                # 2. 【核心逻辑回归】减去初始位置，得到相对位置
                q_real_rel = q_real_abs - self.start_joints
                
                # 3. 乘以方向系数 -> 得到模型需要的 Sim 角度
                q_sim = q_real_rel * HARDWARE_DIR
                
                # 4. 夹爪处理 (夹爪通常是绝对值，或者根据您的习惯也需要相对值？
                # 这里保持绝对值映射，因为夹爪行程是固定的)
                m_grip = self.motors["gripper"]
                self.control.refresh_motor_status(m_grip)
                g_pos = m_grip.getPosition()
                g_norm = (g_pos - self.gripper_open_pos) / (self.gripper_closed_pos - self.gripper_open_pos)
                
                state = np.concatenate([q_sim, [g_norm]]).astype(np.float32)
            except Exception as e:
                # 偶尔读取失败不要崩溃，打印警告即可
                # logger.warning(f"Read sensor failed: {e}")
                pass
            
        obs["state"] = state
        
        # 图像处理
        images = {}
        for name, cap in self.cameras.items():
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    images[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    images[name] = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                images[name] = np.zeros((480, 640, 3), dtype=np.uint8)
        obs["images"] = images
        return obs

    def send_action(self, action: np.ndarray):
        if not self.is_connected: return

        # 1. 解析模型输出 (Sim 坐标系)
        q_sim_target = action[:6]
        g_target = action[6]
        
        # 2. 转换方向 (Sim -> Real Relative)
        q_real_rel_target = q_sim_target * HARDWARE_DIR
        
        # 3. 【核心逻辑回归】加上初始偏移，还原为物理绝对位置
        # 如果模型输出 0，则目标就是 start_joints
        q_real_abs_target = q_real_rel_target + self.start_joints
        
        # 4. 限制与平滑
        q_current_real = self._read_physical_joints_raw()
        q_safe_cmd = []
        
        for i in range(6):
            # 这里的限位其实比较难做，因为不知道 start_joints 是多少
            # 暂时放宽绝对限位，依赖相对运动幅度限制
            
            delta = q_real_abs_target[i] - q_current_real[i]
            # 限制单步最大速度 (Rad)
            delta = np.clip(delta, -self.max_step_rad, self.max_step_rad)
            q_safe_cmd.append(q_current_real[i] + delta)

        DM4340_SPEED = 10.0
        
        # 发送关节指令
        for i in range(6):
            motor = self.motors[f"joint_{i+1}"]
            self.control.control_Pos_Vel(motor, q_safe_cmd[i], DM4340_SPEED)
            
        # 发送夹爪指令
        m_grip = self.motors["gripper"]
        g_real = (g_target * (self.gripper_closed_pos - self.gripper_open_pos)) + self.gripper_open_pos
        self.control.control_Pos_Vel(m_grip, g_real, 10.0)

    def close(self):
        for cap in self.cameras.values(): cap.release()
        if self.is_connected: self.serial_conn.close()