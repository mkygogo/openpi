import time
import numpy as np
import tyro
from dataclasses import dataclass
from mkygogo.mkrobot.env import MKRobotOpenPIEnv
from mkygogo.mkrobot.dataset_loader import EpisodeActionLoader

@dataclass
class Args:
    dataset_root: str = "/home/jr/PI/data/mkrobot_cube_dataset_backup_56" # 数据集路径
    episode_id: int = 5                                  # 回放第几个 episode
    control_hz: float = 30.0                             # 目标控制频率 (Hz)
    robot_port: str = "/dev/ttyACM0"                     # 机械臂串口

def safety_move_to_start(env, target_state, duration=3.0):
    """
    安全复位：将机械臂从当前位置插值移动到 Episode 的起始位置
    """
    print(">>> [SAFETY] Moving robot to start position...")
    
    # 假设 env.get_observation() 返回的字典里有 'state'
    # 注意：这里需要根据你的 Env 具体实现来获取当前关节角度
    # 如果你的 Env 没有直接返回关节角度的接口，可能需要临时加一个
    # 这里假设 env 内部维护了 self.robot
    
    # 获取当前真实状态 (模拟)
    # 这里的实现依赖于你的 MKRobotOpenPIEnv 怎么写的
    # 通常你需要手动读取一次当前关节
    # current_joints = env.robot.get_joint_positions() 
    
    # 由于我看不到 env 源码，这里做一个假设：
    # 我们发送 target_state 作为一个绝对位置命令，让底层驱动慢慢过去
    # 如果你的 action 是 delta，这里需要你手动处理归位逻辑
    
    print(f"Target Start State: {target_state}")
    print("Please manually ensure robot is close to start state or implement auto-homing.")
    time.sleep(1)
    
    # 简单策略：给予 3 秒时间让操作员确认，或者在这里写一段插值逻辑
    # 如果你的 action 是绝对位置控制：
    # env.apply_action(target_state) 
    # time.sleep(3)

def main(args: Args):
    # 1. 初始化 Loader
    loader = EpisodeActionLoader(args.dataset_root, args.episode_id)
    print(f"Loaded Episode {args.episode_id} with {len(loader)} frames.")

    # 2. 初始化环境
    print("Initializing Robot Environment...")
    # prompt 可以随便填，因为我们不经过模型
    env = MKRobotOpenPIEnv(prompt="replay_test", port=args.robot_port)
    
    try:
        # 3. (可选) 安全移动到初始位置
        start_state = loader.get_start_state()
        # 注意：如果你的 Action 是 Delta (速度/增量)，这一步非常重要，
        # 必须先把机械臂摆到和录制时一样的姿态，否则回放出来的轨迹是偏的。
        safety_move_to_start(env, start_state)
        
        input(">>> Press ENTER to start 30Hz Replay...")

        # 4. 30Hz 循环回放
        dt = 1.0 / args.control_hz
        frame_count = 0
        
        while True:
            loop_start = time.time()
            
            # A. 获取数据中的 Action
            action = loader.get_action()
            if action is None:
                print("Episode finished.")
                break
                
            # B. 发送给机械臂
            # 你的 Env 应该负责解析这个 action (无论是 delta 还是 absolute)
            #env.apply_action(action)
            env.apply_action({"action": action})
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Replaying Frame {frame_count}/{len(loader)}")

            # C. 保持 30Hz 频率
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[Warning] Loop lag! Took {elapsed:.4f}s (Target: {dt:.4f}s)")

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main(tyro.cli(Args))