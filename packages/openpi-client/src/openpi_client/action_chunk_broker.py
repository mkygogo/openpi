from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    # @override
    # def infer(self, obs: Dict) -> Dict:  # noqa: UP006
    #     # === 🐛 [DEBUG PROBE] START ===
    #     print(f"\n🔎 [Broker] infer called. cur_step={self._cur_step}, horizon={self._action_horizon}")
    #     # === 🐛 [DEBUG PROBE] END ===
    #     if self._last_results is None:
    #         print("🔎 [Broker] No cached results. Calling policy.infer(obs)...")
    #         self._last_results = self._policy.infer(obs)
    #         self._cur_step = 0

    #         # === 🐛 [DEBUG PROBE] 打印模型返回的原始数据形状 ===
    #         def print_shape(path, x):
    #             if isinstance(x, np.ndarray):
    #                 print(f"🔎 [Broker] RAW MODEL OUTPUT | Path: {path} | Shape: {x.shape} | Ndim: {x.ndim}")
    #                 # 如果是 actions，打印前几个数值看看
    #                 if "actions" in str(path) or path == "actions":
    #                     print(f"   -> First row data: {x.flatten()[:14]}") # 打印前14个数看一眼
    #             else:
    #                 print(f"🔎 [Broker] RAW MODEL OUTPUT | Path: {path} | Type: {type(x)}")
            
    #         print("🔎 [Broker] Inspecting model output structure:")
    #         tree.map_structure_with_path(print_shape, self._last_results)
    #         print("--------------------------------------------------")
    #         # === 🐛 [DEBUG PROBE] END ===

    #     def slicer(x):
    #         if isinstance(x, np.ndarray):
    #             #return x[self._cur_step, ...] 这个是原来的逻辑
    #             # === 🐛 [DEBUG PROBE] 打印切片操作 ===
    #             try:
    #                 # 试图切片前先打印
    #                 # print(f"   -> Slicing {x.shape} at index {self._cur_step}...") 
    #                 val = x[self._cur_step, ...]
    #                 return val
    #             except IndexError as e:
    #                 print(f"\n❌❌❌ [Broker] CRASH DETECTED! ❌❌❌")
    #                 print(f"   Attempted index: {self._cur_step}")
    #                 print(f"   Target array shape: {x.shape}")
    #                 print(f"   Error details: {e}")
    #                 print("   ANALYSIS: If Shape is (7,), it means model returned 1 frame of 7 joints.")
    #                 print("             But we tried to access index > 0 assuming it was time dimension.")
    #                 raise e # 抛出异常让程序停止
    #             # === 🐛 [DEBUG PROBE] END ===
    #         else:
    #             return x

    #     results = tree.map_structure(slicer, self._last_results)
    #     self._cur_step += 1

    #     if self._cur_step >= self._action_horizon:
    #         print(f"🔎 [Broker] Reached horizon {self._action_horizon}. Clearing cache.")
    #         self._last_results = None

    #     return results

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # === 🔍 [DEBUG] 检查是联网还是吃缓存 ===
        if self._last_results is None:
            print(f"🌐 [Broker] 正在联网获取新数据 (Horizon={self._action_horizon})...")
        else:
            # 只有当 step 为 1, 10, 20... 时打印一下，避免刷屏，证明在用缓存
            if self._cur_step % 10 == 0:
                print(f"📦 [Broker] 正在使用本地缓存: 第 {self._cur_step} 帧")
        # ========================================
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0
            
            # --- 🛡️ [安全性修正] 维度归一化 ---
            # 如果 actions 是 (7,)，强制转为 (1, 7)，防止单帧被误判
            def ensure_chunk_dim(path, x):
                if isinstance(x, np.ndarray) and x.ndim == 1 and "actions" in str(path):
                    return x[None, ...]
                return x
            self._last_results = tree.map_structure_with_path(ensure_chunk_dim, self._last_results)

        # 1. 确定 Chunk Size (以 actions 为准)
        chunk_size = 1
        actions = self._last_results.get("actions")
        if actions is not None and isinstance(actions, np.ndarray):
            chunk_size = actions.shape[0]

        # 2. 智能切片函数
        def slicer(path, x):
            if isinstance(x, np.ndarray):
                # 规则 A: 如果是 'actions'，必须切片 (受 horizon 和 chunk_size 限制)
                if "actions" in str(path):
                    # 安全索引：取 min 确保不越界 (虽然逻辑上 cur_step 应该受控)
                    idx = min(self._cur_step, x.shape[0] - 1)
                    return x[idx, ...]
                
                # 规则 B: 如果其他数组的第一维等于 chunk_size，且维度大于1，大概率也是序列，切它
                # (例如 logits: (50, 7) -> 切)
                # (例如 state: (7,) 且 chunk_size=50 -> 不切)
                if x.shape[0] == chunk_size and x.ndim > 1:
                    idx = min(self._cur_step, x.shape[0] - 1)
                    return x[idx, ...]
                
                # 规则 C: 其他情况 (如 state, timing)，保持原样，直接透传
                return x
                
            return x

        # 使用 with_path 以便识别 key
        results = tree.map_structure_with_path(slicer, self._last_results)
        self._cur_step += 1

        # 3. 决定何时获取下一批数据
        # 满足任一条件即刷新：
        # - 达到用户设定的 Horizon (25)
        # - 消耗完了当前 Chunk 的所有数据 (50)
        if self._cur_step >= self._action_horizon or self._cur_step >= chunk_size:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
