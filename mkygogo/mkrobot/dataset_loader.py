import pandas as pd
import numpy as np
import os

class EpisodeActionLoader:
    def __init__(self, dataset_root, episode_id, chunk_id=0):
        """
        加载指定 Episode 的 Action 数据
        """
        self.parquet_path = os.path.join(
            dataset_root, 
            "data", 
            f"chunk-{chunk_id:03d}", 
            f"episode_{episode_id:06d}.parquet"
        )
        
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Dataset file not found: {self.parquet_path}")
            
        print(f"[Loader] Loading actions from: {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)
        self.total_frames = len(self.df)
        self.current_idx = 0

    def __len__(self):
        return self.total_frames

    def get_action(self):
        """
        每次调用返回下一帧的 Action，如果没有了返回 None
        """
        if self.current_idx >= self.total_frames:
            return None
        
        # 获取当前行的 action
        action = self.df.iloc[self.current_idx]['action']
        
        # 确保转为 numpy 数组 (float32)
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
            
        self.current_idx += 1
        return action

    def get_start_state(self):
        """
        获取该 Episode 第一帧的机械臂状态 (observation.state)
        用于在开始回放前，先把机械臂移动到初始位置，防止瞬移
        """
        state = self.df.iloc[0]['observation.state']
        return np.array(state, dtype=np.float32)
        
    def reset(self):
        self.current_idx = 0