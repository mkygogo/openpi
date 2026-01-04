import logging
import time
import tyro
import dataclasses
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from openpi_client.runtime import runtime
from openpi_client.runtime.agents import policy_agent

from mkygogo.mkrobot.env import MKRobotOpenPIEnv

# =============================================================================
# 🔧 配置区域
# =============================================================================
# 反归一化参数 (必须与录数据时一致)
JOINT_NORM_SCALE = np.array([3.0, 3.0, 3.0, 1.7, 0.4, 2.0, 1.0], dtype=np.float32)

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    prompt: str = "pick up the small cube and place it in the box"
    
    # [关键修改] 加大到 30，解决"一顿一顿"的问题
    # 30Hz 下，30帧 = 1秒缓存，足够抵抗网络波动
    action_horizon: int = 30  
    
    control_hz: float = 30.0  

# =============================================================================
# 🛠️ 包装类：负责反归一化 + 维度安全检查 (核心修改)
# =============================================================================
# class NormalizedMKRobotEnv(MKRobotOpenPIEnv):
#     def apply_action(self, action: dict):
#         if "actions" in action:
#             # 1. 获取原始数据
#             raw_action = np.array(action["actions"], dtype=np.float32)
            
#             # --- 🛡️ [关键修复] 维度安全卫士 ---
#             # 问题根源：如果 horizon 很大，raw_action 可能会变成 (1, 7) 甚至 (N, 7)
#             # 这会导致底层驱动在用 raw_action[i] 访问时，访问的是"第i行"而不是"第i个关节"
#             # 进而导致 IndexError: index 7 out of bounds
            
#             # 强制拍扁成一维数组 (7,)
#             raw_action = raw_action.flatten()
            
#             # 截断/检查：确保只有 7 个数
#             if raw_action.shape[0] > 7:
#                 # 如果传来了多个动作，我们只取第一个 (通常 Runtime 会帮我们切好，但为了保险)
#                 raw_action = raw_action[:7]
#             elif raw_action.shape[0] < 7:
#                 print(f"Error: Action shape mismatch! Expected 7, got {raw_action.shape}")
#                 return
#             # -----------------------------------
            
#             # 2. 反归一化
#             physical_action = raw_action * JOINT_NORM_SCALE
            
#             # 3. 更新回去
#             action["actions"] = physical_action
            
#         super().apply_action(action)
class NormalizedMKRobotEnv(MKRobotOpenPIEnv):
    def apply_action(self, action: dict):
        if "actions" in action:
            # 1. 获取原始数据 (可能形状是 (30, 7) 或者 (1, 30, 7))
            raw_action = np.array(action["actions"], dtype=np.float32)
            raw = np.array(action["actions"])
            #print(f"\n🐛 [Main] Wrapper received action shape: {raw.shape}")
            # --- 🛡️ [修正] 删除破坏维度的代码 ---
            # ❌ 不要 raw_action.flatten() -> 这会把30帧拍扁成一长条
            # ❌ 不要 raw_action[:7] -> 这会丢掉后29帧，导致卡顿
            
            # 2. 维度安全处理 (确保最后是 (N, 7))
            # 这里的目的是让 numpy 广播乘法能正常工作
            if raw_action.ndim == 3: # 处理 (1, 30, 7) 这种情况
                raw_action = raw_action[0] 
            #print(f"🐛 [Main] After squeeze: {raw_action.shape}")
            # 此时 raw_action 应该是 (30, 7) 或 (1, 7)
            
            # 3. 反归一化 (保留！幅度就靠它了)
            # numpy 会自动把 (7,) 的系数广播到每一帧上
            physical_action = raw_action * JOINT_NORM_SCALE
            
            # 4. 更新回去
            action["actions"] = physical_action
            
        # 5. 交给父类 (env.py) 去逐帧执行
        super().apply_action(action)
# =============================================================================

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    
    logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
    try:
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port
        )
        server_meta = policy_client.get_server_metadata()
        logging.info(f"Connected! Server metadata: {server_meta}")
    except Exception as e:
        logging.error(f"Failed to connect: {e}")
        return

    try:
        logging.info("Initializing MKRobot environment...")
        # 使用这一层 Wrapper
        env = NormalizedMKRobotEnv(prompt=args.prompt, port="/dev/ttyACM0")
    except Exception as e:
        logging.error(f"Hardware init failed: {e}")
        return

    agent = policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=policy_client,
            action_horizon=args.action_horizon, 
        )
    )

    rt = runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[],
        max_hz=args.control_hz,
        num_episodes=1000 
    )

    try:
        logging.info(f"Starting inference at {args.control_hz}Hz with Horizon={args.action_horizon}...")
        print(">>> 机械臂运行中，按 Ctrl+C 停止 <<<")
        rt.run()
        
    except KeyboardInterrupt:
        logging.info("Stopping...")
        #给归位过程加锁，防止二次中断报错
        try:
            # 创建一个全 0 的姿态作为归位目标 (根据你的机器人实际情况调整)
            # 注意：这里假设你的归位就是回到 0 位。如果不是，请保留你原有的逻辑
            target_state = np.zeros(7, dtype=np.float32)
            
            # 强制执行归位，并忽略期间的任何按键中断
            env.controller.apply_action(target_state)
            
        except Exception:
            # 如果在归位时用户又狂按 Ctrl+C，或者再次触发检测，直接忽略
            pass
        print("✅ 已归位。")
        
        
    except Exception as e:
        logging.error(f"Runtime error?: {e}")

    finally:
        # 安全归位逻辑
        if 'env' in locals():
            print("\n🛑 正在安全归位...")
            try:
                # 简单归位：给一个全 0 的动作（或者你的归位逻辑）
                # 注意：这里构造一个 (1, 7) 的动作，而不是 (7,)
                target_state = np.zeros(7, dtype=np.float32)
                # 如果你想保持夹爪状态，可以读取当前状态（略）
                
                # 缓慢归位循环
                for _ in range(50):
                   env.controller.apply_action(target_state)
                   time.sleep(0.02)
                   
                print("✅ 已归位。")
            except Exception as e:
                print(f"归位失败: {e}")
            env.close()            

if __name__ == "__main__":
    main(tyro.cli(Args))