import sys
import tty
import termios
import time
import numpy as np
import threading
import cv2
import queue

# 引用驱动
try:
    from mkygogo.mkrobot.hardware.mk_driver import MKRobotStandalone, JOINT_LIMITS
except ImportError:
    print("❌ 错误：找不到驱动模块。请确保在项目根目录运行。")
    sys.exit(1)

# ================= 配置区域 =================
# 请根据你的实际情况修改摄像头索引
# 通常 0 是电脑自带摄像头，2, 4... 是外接 USB 摄像头
CAMERA_CONFIG = {
    'top':   {'index': 0, 'width': 640, 'height': 480},
    'wrist': {'index': 2, 'width': 640, 'height': 360}
}
# ===========================================

# 全局变量，用于线程间通信
latest_obs = None
running = True
obs_lock = threading.Lock()

# ==========================================
# 键盘输入处理 (非阻塞)
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
        ch = sys.stdin.read(1)
        if ch == '\x1b':
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
# 摄像头显示线程
# ==========================================
def camera_loop(robot):
    global latest_obs, running
    print("📸 摄像头线程已启动...")
    
    while running:
        # 1. 获取观测数据 (包含图像和关节状态)
        # 注意：get_observation 会读取摄像头，所以只能在一个线程调用
        obs = robot.get_observation()
        
        # 2. 更新全局状态 (加锁)
        with obs_lock:
            latest_obs = obs
        
        # 3. 显示图像
        images = obs.get('images', {})
        if images:
            for name, img in images.items():
                if img is not None and img.size > 0:
                    # OpenCV 默认是 BGR，如果驱动返回的是 RGB，需要转换
                    # 假设驱动返回的是 RGB (根据之前的代码)
                    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Camera: {name}", bgr_img)
        
        # 4. 响应 GUI 按键 (按 'q' 退出)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()
    print("📸 摄像头线程停止")

# ==========================================
# 主程序
# ==========================================
def main():
    global running, latest_obs
    
    print("正在连接机械臂和摄像头...")
    robot = MKRobotStandalone(
        port="/dev/ttyACM0", 
        joint_velocity_scaling=1.0,
        camera_indices=CAMERA_CONFIG  # 传入摄像头配置
    )
    
    try:
        robot.connect()
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 启动显示线程
    display_thread = threading.Thread(target=camera_loop, args=(robot,))
    display_thread.start()

    # 等待第一帧数据
    print("等待摄像头画面...")
    while latest_obs is None and running:
        time.sleep(0.1)

    if not running:
        return

    # 初始化目标位置
    with obs_lock:
        target_pos = latest_obs["state"][:7].copy().astype(np.float32)

    JOINT_STEP = 0.05
    GRIPPER_STEP = 0.1

    print("\n" + "="*50)
    print(" 🎮 机械臂视觉遥操作 (Test Robot Cam)")
    print("="*50)
    print(" [1-6] 选择关节 J1-J6")
    print(" [7]   选择夹爪")
    print(" [ESC] 退出程序")
    print(" 提示: 选中窗口按 'q' 也可以退出")
    print("="*50)

    try:
        with KeyboardInput() as kb:
            while running:
                # 显示状态
                with obs_lock:
                    curr_real_pos = latest_obs["state"][:7]
                
                print(f"\r\n当前真实值: {np.round(curr_real_pos, 3)}")
                print("等待选择电机 (1-7) 或 ESC退出: ", end='', flush=True)
                
                key = kb.get_key()

                if key == 'ESC' or not running:
                    break
                
                if key in ['1', '2', '3', '4', '5', '6', '7']:
                    motor_idx = int(key) - 1
                    motor_name = f"Joint {key}" if key != '7' else "Gripper"
                    
                    print(f"\n\n>>> 已选中: {motor_name}")
                    print(" [↑] 增加/闭合  [↓] 减小/张开  [q] 返回")
                    
                    while running:
                        # 每次循环都刷新一下目标值基准，防止误差累积
                        # 但为了控制平滑，这里我们保持 target_pos 独立
                        print(f"\r{motor_name} 目标: {target_pos[motor_idx]:.3f}   ", end='', flush=True)
                        
                        # 非阻塞检查按键 (这里简化为阻塞读取，因为图像在另一线程)
                        cmd_key = kb.get_key()
                        
                        if cmd_key == 'q':
                            break
                        if cmd_key == 'ESC':
                            running = False
                            break
                        
                        # 计算新目标
                        new_val = target_pos[motor_idx]
                        step = GRIPPER_STEP if motor_idx == 6 else JOINT_STEP
                        
                        if cmd_key == 'UP':
                            new_val += step
                        elif cmd_key == 'DOWN':
                            new_val -= step
                        else:
                            continue

                        # 限位检查
                        if motor_idx in JOINT_LIMITS:
                            min_l, max_l = JOINT_LIMITS[motor_idx]
                            new_val = np.clip(new_val, min_l, max_l)
                        elif motor_idx == 6:
                            new_val = np.clip(new_val, 0.0, 1.0)
                        
                        target_pos[motor_idx] = new_val
                        
                        # 发送动作
                        robot.send_action(target_pos)
                        time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        running = False
        print("\n正在停止...")
        display_thread.join() # 等待显示线程结束
        try:
            robot.disconnect()
        except:
            pass
        print("Done.")

if __name__ == "__main__":
    main()