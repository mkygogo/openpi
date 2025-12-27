import time
import numpy as np
import logging
import sys
import os

# 确保能引用到 mkygogo 包
sys.path.append(os.getcwd())

from mkygogo.mkrobot.hardware.mk_driver import MKRobotStandalone

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestRobot")

def test_hardware():
    print("="*50)
    print(" 🛠️  MKRobot 硬件连接测试")
    print("="*50)

    # 1. 尝试初始化
    port = "/dev/ttyACM0"  # 如果不确定，可以改为 /dev/ttyUSB0 试试
    print(f"1. 正在尝试连接串口: {port} ...")
    
    try:
        # 只测试连接，不指定相机，排除相机干扰
        robot = MKRobotStandalone(port=port, camera_indices={})
        robot.connect()
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("💡 建议检查: 1. 串口权限 (sudo chmod 666 /dev/ttyACM0)  2. 串口号是否正确")
        return

    if not robot.is_connected:
        print("❌ 串口已打开，但无法与机械臂建立通信。")
        print("💡 建议检查: 24V/48V 电源是否开启？急停开关是否按下？")
        return

    print("✅ 机械臂连接成功！")

    # 2. 读取状态
    print("\n2. 正在读取关节状态 (持续 5 秒)...")
    try:
        for i in range(50):
            obs = robot.get_observation()
            state = obs["state"] # 7维数据
            # 打印格式化后的角度
            state_str = ", ".join([f"{x:5.2f}" for x in state])
            print(f"\r[{i}] 关节角度: [{state_str}]", end="")
            time.sleep(0.1)
        print("\n✅ 读取测试完成。")
        
    except Exception as e:
        print(f"\n❌ 读取过程出错: {e}")

    # 3. 简单的微动测试 (可选，非常轻微的动作)
    print("\n3. 准备进行微动测试 (警告: 机械臂将轻微抖动)")
    confirm = input("输入 'y' 继续，其他键跳过: ")
    if confirm.lower() == 'y':
        try:
            current_joints = robot._read_physical_joints()
            print(f"当前物理角度: {current_joints}")
            
            # 目标：在当前位置上给第一个关节 +0.05 弧度
            target_joints = current_joints.copy()
            target_joints[0] += 0.05 
            
            # 发送指令 (注意：MKRobotStandalone.send_action 接受的是 Simulation 坐标系数据)
            # 为了安全，这里我们直接调用底层的 control 接口测试一下物理运动
            # 或者简单跳过，因为上面的读取成功通常意味着通信正常
            print("为安全起见，本次测试仅验证读取功能。如果读数正常变动，说明通信是好的。")
            
        except Exception as e:
            print(f"❌ 运动测试出错: {e}")

    robot.close()
    print("\n🎉 测试结束")

if __name__ == "__main__":
    test_hardware()