import sys
import select
import termios
import tty
import logging
import numpy as np
import time
from typing import Dict

from .hardware.mk_driver import MKRobotStandalone

logger = logging.getLogger("MKController")

class MKController:
    def __init__(self, port="/dev/ttyACM0", camera_indices=None):
        self.driver = MKRobotStandalone(port=port, camera_indices=camera_indices)
        self.is_paused = False
        
    def connect(self):
        self.driver.connect()
        # 设置终端为非规范模式以捕获按键 (仅 Linux)
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print("\n" + "="*40)
        print(" 🎮 控制器就绪")
        print(" [SPACE] : 紧急归零 (Home)")
        print(" [Q]     : 退出")
        print("="*40 + "\n")

    def check_user_input(self):
        """非阻塞检查按键"""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == ' ': # 空格键归零
                self.is_paused = not self.is_paused
                if self.is_paused:
                    logger.warning("\n>>> ⏸️  已暂停! 正在归零... (再次按空格恢复) <<<")
                    self.go_home()
                else:
                    logger.warning("\n>>> ▶️  恢复运行! <<<")
            elif key.lower() == 'q':
                logger.info("用户请求退出")
                raise KeyboardInterrupt
            
    def go_home(self):
        """强制回到零位 (即上电位置)"""
        logger.info("Executing Home Sequence...")
        
        # 零位对应的是：所有关节 Sim 角度为 0
        home_action = np.zeros(7, dtype=np.float32)
        # 夹爪可能需要打开
        home_action[6] = 0.0 
        
        # 慢速发送几次指令，确保归位
        for _ in range(20):
            self.driver.send_action(home_action)
            time.sleep(0.05)
            
        logger.info("Home Sequence Complete.")

    def get_observation(self):
        return self.driver.get_observation()

    def apply_action(self, action: np.ndarray):
        # 1. 检查是否有用户按键
        self.check_user_input()
        
        # 2. 如果正在归零中，忽略模型指令
        if self.is_paused:
            return

        # 3. 正常执行模型指令
        self.driver.send_action(action)

    def close(self):
        # 恢复终端设置
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.driver.close()