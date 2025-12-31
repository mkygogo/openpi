import time
import numpy as np
from mkygogo.mkrobot.mk_controller import MKController

def main():
    print("=== 关节减速比校准测试 ===")
    print("注意：机械臂即将让第1个关节 (底座) 转动。请确保周围没有障碍物。")
    print("目标：转动 0.5 弧度 (约 28.6 度)")
    
    controller = MKController(port="/dev/ttyACM0") # 请确认端口
    try:
        controller.connect()
        input(">>> 按回车键开始 (手请放在急停开关/电源键上)...")

        # 1. 读取当前位置
        obs = controller.get_observation()
        start_j1 = obs["state"][0]
        print(f"初始 J1 位置: {start_j1:.4f} rad")

        # 2. 目标位置 (缓慢增加 0.5)
        target_j1 = start_j1 + 0.5
        
        # 3. 缓慢执行 (耗时 3 秒)
        steps = 100
        for i in range(steps):
            # 线性插值计算当前目标
            alpha = (i + 1) / steps
            current_target = start_j1 + (0.5 * alpha)
            
            # 构造全套动作 (其他关节保持不动)
            # 注意：这里我们假设 get_observation 返回的是当前真实的 Sim 角度
            # 我们只需要把第0个分量改成新的目标
            action = obs["state"][:7].copy() # 复制当前状态作为基础
            action[0] = current_target       # 只修改 J1
            
            controller.apply_action(action)
            time.sleep(0.03) # 30Hz

        print(f"指令发送完毕。目标 J1 增量: +0.5 rad (+28.6度)")
        print("请观察机械臂实际转动了多少？")
        print("A. 大约 30 度 (像钟表走了5分钟的刻度) -> 正常，无需缩放")
        print("B. 大约 3 度 (几乎没动) -> 减速比问题 (约为 10:1)")
        
    except KeyboardInterrupt:
        print("停止")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        controller.close()

if __name__ == "__main__":
    main()