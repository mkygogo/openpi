import time
import numpy as np
import tyro
from dataclasses import dataclass
from mkygogo.mkrobot.env import MKRobotOpenPIEnv
from mkygogo.mkrobot.dataset_loader import EpisodeActionLoader

@dataclass
class Args:
    dataset_root: str = "/home/jr/PI/data/mkrobot_cube_dataset_backup_56"
    episode_id: int = 5
    control_hz: float = 30.0
    robot_port: str = "/dev/ttyACM0"

# 【关键】这是你采集代码里的归一化参数
# J1-J3=3.0, J4=1.7, J5=0.4, J6=2.0, Gripper=1.0
JOINT_NORM_SCALE = np.array([3.0, 3.0, 3.0, 1.7, 0.4, 2.0, 1.0], dtype=np.float32)

def safety_move_to_start(env, target_state):
    print("\n>>> [重要提示] 请手动协助机械臂归位 <<<")
    
    # 这里也要反归一化打印，否则提示的位置也是缩小的
    real_target = target_state * JOINT_NORM_SCALE
    
    print(f"该 Episode 录制时的初始关节角度 (Sim Frame, 已反归一化):")
    print(np.round(real_target, 4))
    print("\n请在回车前，手动将机械臂摆成大致相似的姿态。")

def main(args: Args):
    # 1. 加载数据
    loader = EpisodeActionLoader(args.dataset_root, args.episode_id)
    print(f"Loaded Episode {args.episode_id} with {len(loader)} frames.")

    # 2. 初始化环境
    print("Initializing Robot Environment...")
    env = MKRobotOpenPIEnv(prompt="replay_test", port=args.robot_port)
    
    try:
        # 3. 准备工作
        # 获取第一帧动作作为初始位置
        start_action_norm = loader.get_action() # 这是归一化的
        # 重置loader指针（如果loader没有peek功能，这里假设get_start_state是独立的）
        # 为了保险，我们重新初始化loader或者假设loader.get_start_state()是正确的
        # 既然你之前的代码用了 loader.get_start_state()，我们继续用
        start_state_norm = loader.get_start_state()
        
        safety_move_to_start(env, start_state_norm)
        
        input(">>> 确认安全后，按回车键开始 30Hz 原速回放...")

        dt = 1.0 / args.control_hz
        frame_count = 0
        
        while True:
            loop_start = time.time()
            
            # 获取观测
            obs = env.get_observation()
            
            # A. 获取数据 (归一化的)
            action_norm = loader.get_action()
            if action_norm is None:
                print("Episode finished.")
                break
            
            # ==========================================================
            # 🔑【核心修复】反归一化 (Un-normalize)
            # Real_Pos = Norm_Pos * Scale
            # ==========================================================
            action_real = action_norm * JOINT_NORM_SCALE
            # ==========================================================

            # --- 🔍 诊断打印 (使用反归一化后的真实值) ---
            if frame_count % 30 == 0 and obs is not None:
                try:
                    curr_state = obs.get("state")
                    if curr_state is not None:
                        print(f"\n--- Frame {frame_count} 诊断 ---")
                        print(f"{'关节':<5} | {'目标(Real)':<10} | {'实际(Obs)':<10} | {'偏差':<8}")
                        for j in range(6):
                            t = action_real[j]
                            c = curr_state[j]
                            diff = t - c
                            mark = "(!)" if abs(diff) > 0.1 else ""
                            print(f"J{j+1:<5} | {t:<10.3f} | {c:<10.3f} | {diff:<8.3f} {mark}")
                except: pass
            # ----------------------------------------

            # B. 发送指令 (发送真实物理角度)
            env.apply_action({"actions": action_real})
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Replaying Frame {frame_count}/{len(loader)}", end='\r')

            # C. 控频
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # --- 安全归位 ---
        print("\n🛑 正在执行安全归位 (Go Home)...")
        try:
            obs = env.get_observation()
            if obs is not None and "state" in obs:
                current = obs["state"]
                target = np.zeros_like(current)
                target[6] = current[6] 
                
                for i in range(100):
                    alpha = (i + 1) / 100.0
                    interp = current * (1 - alpha) + target * alpha
                    env.apply_action({"actions": interp})
                    time.sleep(0.02)
                print("✅ 已归位。")
        except: pass

        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main(tyro.cli(Args))