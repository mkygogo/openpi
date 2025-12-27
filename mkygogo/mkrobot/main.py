import logging
import time
import tyro
import dataclasses
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from openpi_client.runtime import runtime
from openpi_client.runtime.agents import policy_agent

# 确保这里的引用路径正确
# 前提：你的 mkygogo 目录下有 __init__.py，且在 openpi 根目录下运行
from mkygogo.mkrobot.env import MKRobotOpenPIEnv

@dataclasses.dataclass
class Args:
    # Server 地址（本地跑就是 localhost，远程跑填 IP）
    host: str = "localhost"
    port: int = 8000
    
    # 提示词，需要与训练任务匹配
    prompt: str = "pick up the small cube and place it in the box"
    
    # 动作执行参数
    # Pi0 模型通常一次输出 50-100 步动作
    # ActionChunkBroker 会帮你切分并平滑执行
    action_horizon: int = 20  
    
    # 控制频率 (Hz)
    # 建议从 10Hz 开始调试，MKRobot 的 USB 读取速度通常在 15-30Hz 之间
    control_hz: float = 10.0  

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    
    # 1. 连接推理服务器 (WebSocket)
    logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
    try:
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port
        )
        # 获取元数据，确认连接成功
        server_meta = policy_client.get_server_metadata()
        logging.info(f"Connected! Server metadata: {server_meta}")
    except Exception as e:
        logging.error(f"Failed to connect to server: {e}")
        logging.error("Hint: Please make sure 'start_server.sh' is running and the port matches.")
        return

    # 2. 初始化硬件环境
    # port 参数对应你的机械臂串口
    try:
        logging.info("Initializing MKRobot environment...")
        env = MKRobotOpenPIEnv(prompt=args.prompt, port="/dev/ttyACM0")
    except Exception as e:
        logging.error(f"Failed to initialize robot hardware: {e}")
        return

    # 3. 构建 Agent
    # ActionChunkBroker 是 OpenPI 客户端的核心组件
    # 它负责维护一个动作缓冲区，处理模型输出的长序列动作 (Chunk)
    agent = policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=policy_client,
            action_horizon=args.action_horizon, 
        )
    )

    # 4. 运行主循环 (Runtime Loop)
    # Runtime 会按照 control_hz 的频率调用:
    # env.get_observation() -> agent.get_action() -> env.apply_action()
    rt = runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[],
        max_hz=args.control_hz,
        num_episodes=1000 # 持续运行，直到手动停止
    )

    try:
        logging.info(f"Starting inference loop at {args.control_hz}Hz... Press Ctrl+C to stop.")
        rt.run()
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        # 确保退出时释放硬件资源
        if 'env' in locals():
            env.close()
        logging.info("Exited cleanly.")

if __name__ == "__main__":
    main(tyro.cli(Args))