import logging
import time
import tyro
import dataclasses
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
#from openpi_client.runtime import runtime
#from openpi_client.runtime.agents import policy_agent

from mkygogo.mkrobot.env import MKRobotOpenPIEnv
from mkygogo.mkrobot.mk_controller import RobotResetException

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

    #初始化 Broker
    if args.action_horizon > 1:
        policy_client = action_chunk_broker.ActionChunkBroker(
            policy=policy_client, 
            action_horizon=args.action_horizon
        )

    logging.info(f"Starting inference at {args.control_hz}Hz with Horizon={args.action_horizon}...")
    print(f"\n>>> 提示词: {args.prompt} <<<")
    print(">>> 机械臂运行中，按 Ctrl+C 停止，按空格键暂停/重置 <<<\n")
    try:
        while True:
            # === 单次任务循环 (处理 Space 重置) ===
            try:
                logging.info("Starting episode...")
                
                # 1. 重置环境和策略 (清空缓存)
                obs = env.reset()
                policy_client.reset() 

                # 2. 推理循环
                while not env.is_episode_complete():
                    # 网络推理
                    action = policy_client.infer(obs)
                    
                    # 执行动作 
                    # (如果用户按了 Space，这里会抛出 RobotResetException)
                    env.apply_action(action)
                    
                    # 获取新观测
                    obs = env.get_observation()

            # 3. 捕获重置信号
            except RobotResetException:
                print("\n⚠️  [System] 收到重置请求，丢弃当前任务，重新开始...\n")
                # 跳回 while True 开头，触发下一次 reset()
                continue
        
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