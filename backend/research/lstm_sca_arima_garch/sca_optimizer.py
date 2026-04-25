"""Sine Cosine Algorithm (SCA) — 用於優化 LSTM/ARIMA 超參數。

論文：Mirjalili, S. (2016). SCA: A Sine Cosine Algorithm for solving optimization problems.
直觀理解：N 個 agent 在搜尋空間移動，sin/cos 函式控制朝向最佳解或遠離最差解。

PoC 階段不啟用（避免長時間訓練），完整啟用見 hybrid_pipeline.py 的註解。
"""

import math
import random
from typing import Callable


def sca_optimize(
    objective: Callable[[list[float]], float],
    bounds: list[tuple[float, float]],
    n_agents: int = 10,
    max_iter: int = 30,
    seed: int | None = None,
) -> tuple[list[float], float]:
    """SCA 主迴圈，回傳 (best_position, best_score)。

    Args:
        objective: 目標函式（給 position vector，回傳要最小化的 score）
        bounds: 每個維度的 (low, high) 範圍
        n_agents: 搜尋代理數量
        max_iter: 最大迭代次數
        seed: 隨機種子（可重現）

    Returns:
        (best_position, best_score)
    """
    if seed is not None:
        random.seed(seed)

    dim = len(bounds)
    # 初始化 agents（隨機位置）
    agents = [
        [random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n_agents)
    ]
    scores = [objective(a) for a in agents]
    best_idx = scores.index(min(scores))
    best_pos = list(agents[best_idx])
    best_score = scores[best_idx]

    a_const = 2.0  # SCA 論文推薦：r1 在 [0, a] 線性遞減

    for t in range(max_iter):
        r1 = a_const - t * (a_const / max_iter)  # 線性遞減
        for i in range(n_agents):
            for j in range(dim):
                r2 = random.uniform(0, 2 * math.pi)
                r3 = random.uniform(0, 2)
                r4 = random.random()
                if r4 < 0.5:
                    new_val = agents[i][j] + r1 * math.sin(r2) * abs(r3 * best_pos[j] - agents[i][j])
                else:
                    new_val = agents[i][j] + r1 * math.cos(r2) * abs(r3 * best_pos[j] - agents[i][j])
                # Clip to bounds
                agents[i][j] = max(bounds[j][0], min(bounds[j][1], new_val))

            scores[i] = objective(agents[i])
            if scores[i] < best_score:
                best_score = scores[i]
                best_pos = list(agents[i])

    return best_pos, best_score
