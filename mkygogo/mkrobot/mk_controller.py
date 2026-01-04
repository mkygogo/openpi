import time
import logging
import numpy as np
import sys
import select
import tty
import termios
from typing import Dict, Any, Optional

# 导入底层驱动
from mkygogo.mkrobot.hardware.mk_driver import MKRobotStandalone

logger = logging.getLogger(__name__)

# === 🌟 [新增] 定义重置异常 ===
class RobotResetException(Exception):
    """用户请求重置环境（通常通过按空格键恢复后触发）"""
    pass

class MKController:
    """
    负责处理用户输入、安全检查，并将高层动作转发给底层驱动。
    """
    def __init__(self, port: str = "/dev/ttyACM0", camera_indices: Dict[str, int] = None):
        self.driver = MKRobotStandalone(port=port, camera_indices=camera_indices)
        self.is_connected = False
        self.old_settings = termios.tcgetattr(sys.stdin)

    def connect(self):
        try:
            self.driver.connect()
            self.is_connected = True
            tty.setcbreak(sys.stdin.fileno())
            
            print("\n========================================")
            print(" 🎮 控制器就绪")
            print(" [SPACE] : 暂停并归零 (再次按 SPACE 重置推理)")
            print(" [Q]     : 退出")
            print("========================================\n")
            
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            self.restore_terminal()
            sys.exit(1)

    def restore_terminal(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_user_input(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key.lower() == 'q':
                logger.info("用户请求退出")
                self.close()
                raise KeyboardInterrupt
            elif key == ' ':
                # 1. 先暂停并归位
                logger.warning("\n>>> ⏸️  已暂停! 正在归零... (再次按空格键 -> 重置推理) <<<")
                self.perform_home_sequence()
                
                # 2. 死循环等待用户再次指令
                while True:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        k = sys.stdin.read(1)
                        if k == ' ':
                            print(">>> 🔄 检测到重置信号，正在重启推理... <<<")
                            # 🌟 [关键] 这里不 break，而是直接抛出异常！
                            # 这会像中断一样，直接炸断 env.apply_action 的循环
                            raise RobotResetException()
                        elif k.lower() == 'q':
                            raise KeyboardInterrupt

    def perform_home_sequence(self):
        logger.info("Executing Home Sequence...")
        home_pos = np.zeros(7, dtype=np.float32)
        for _ in range(40): # 稍微慢一点归位
            self.driver.send_action(home_pos)
            time.sleep(0.033)
        logger.info("Home Sequence Complete.")

    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            return {}
        return self.driver.get_observation()

    def apply_action(self, action: np.ndarray):
        if not self.is_connected:
            return

        # 检查按键 (如果这里抛出 RobotResetException，下面的 send_action 就不会执行)
        self.check_user_input()

        # 发送动作
        self.driver.send_action(action)

    def close(self):
        self.restore_terminal()
        if self.is_connected:
            self.driver.close()
            self.is_connected = False