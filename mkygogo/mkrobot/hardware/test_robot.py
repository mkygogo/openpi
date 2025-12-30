import sys
import tty
import termios
import time
import numpy as np
import os

# 引用你的驱动类
# 注意：请确保你在项目根目录下运行，且 mkygogo 包结构正确
try:
    from mkygogo.mkrobot.hardware.mk_driver import MKRobotStandalone, JOINT_LIMITS
except ImportError:
    print("❌ 错误：找不到驱动模块。请确保在项目根目录(包含mkygogo的目录)下运行此脚本。")
    print("运行命令示例: uv run python test_robot.py")
    sys.exit(1)

# ==========================================
# 键盘输入处理 (Linux Terminal)
# ==========================================
class KeyboardInput:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setraw(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        """读取按键，支持识别箭头键和ESC"""
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # ESC 或 转义序列起始
            # 设置非阻塞读取来检查是否有后续字符
            old_settings = termios.tcgetattr(self.fd)
            new_settings = termios.tcgetattr(self.fd)
            new_settings[6][termios.VMIN] = 0
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSADRAIN, new_settings)
            
            try:
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == '[':
                    if ch3 == 'A': return 'UP'
                    if ch3 == 'B': return 'DOWN'
                return 'ESC'
            finally:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, old_settings)
        return ch

# ==========================================
# 机器人控制逻辑
# ==========================================
def main():
    print("正在连接机械臂...")
    # 初始化机器人 (请确认端口号)
    robot = MKRobotStandalone(port="/dev/ttyACM0", joint_velocity_scaling=1.0)
    
    try:
        robot.connect()
        print("✅ 机械臂连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 1. 初始化目标位置为当前实际位置 (防止瞬移)
    obs = robot.get_observation()
    # 确保是 float32 数组
    target_pos = obs["state"].copy().astype(np.float32)
    
    # 定义步长
    JOINT_STEP = 0.05  # 关节每次调整幅度 (rad)
    GRIPPER_STEP = 0.1 # 夹爪每次调整幅度 (0-1)

    print("\n" + "="*50)
    print(" 🎮 机械臂键盘控制器 (Test Robot)")
    print("="*50)
    print(" [1-6] 选择关节 J1-J6")
    print(" [7]   选择夹爪")
    print(" [ESC] 退出程序")
    print("="*50)

    try:
        with KeyboardInput() as kb:
            while True:
                # --- 主菜单循环 ---
                print(f"\r\n当前状态: {np.round(target_pos, 3)}")
                print("等待选择电机 (1-7) 或 ESC退出: ", end='', flush=True)
                
                key = kb.get_key()

                if key == 'ESC':
                    print("\n正在退出...")
                    break
                
                if key in ['1', '2', '3', '4', '5', '6', '7']:
                    motor_idx = int(key) - 1
                    motor_name = f"Joint {key}" if key != '7' else "Gripper"
                    
                    print(f"\n\n>>> 已选中: {motor_name}")
                    print(" [↑] 增加角度/闭合  [↓] 减小角度/张开  [q] 返回上级")
                    
                    # --- 单电机控制循环 ---
                    while True:
                        current_val = target_pos[motor_idx]
                        
                        # 实时显示当前值
                        print(f"\r{motor_name} 目标值: {current_val:.3f}   ", end='', flush=True)
                        
                        cmd_key = kb.get_key()
                        
                        if cmd_key == 'q':
                            break
                        
                        # 计算新目标
                        new_val = current_val
                        step = GRIPPER_STEP if motor_idx == 6 else JOINT_STEP
                        
                        if cmd_key == 'UP':
                            new_val += step
                        elif cmd_key == 'DOWN':
                            new_val -= step
                        else:
                            continue # 忽略其他按键

                        # 限位检查 (Software Limit)
                        if motor_idx in JOINT_LIMITS:
                            min_l, max_l = JOINT_LIMITS[motor_idx]
                            new_val = np.clip(new_val, min_l, max_l)
                        elif motor_idx == 6: # 夹爪
                            new_val = np.clip(new_val, 0.0, 1.0)
                        
                        # 更新全局目标数组
                        target_pos[motor_idx] = new_val
                        
                        # 发送动作 (发送包含所有关节的完整数组)
                        robot.send_action(target_pos)
                        
                        # 稍微延时，避免发送太快
                        time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n用户强制中断")
    except Exception as e:
        print(f"\n运行时错误: {e}")
    finally:
        print("\n正在断开连接...")
        try:
            robot.disconnect()
        except:
            pass
        print("Done.")

if __name__ == "__main__":
    main()