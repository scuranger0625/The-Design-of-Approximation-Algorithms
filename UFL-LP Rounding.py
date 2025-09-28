# -*- coding: utf-8 -*-
"""
UFL (Uncapacitated Facility Location) — LP Rounding Demo
-------------------------------------------------------
示範流程：
1)（可選）用 LP 鬆弛求出分數解 (x_ij, y_i)
2) 設定常數 c (>1)，計算每個客戶的門檻距離 D_j = c * sum_i d_ij x_ij
3) 定義候選集合 F_j = { i | x_ij > 0 且 d_ij <= D_j }
4) 依 D_j 非遞減排序處理客戶（Challenge 2）
   - 若 F_j 與已連線客戶 k 的 F_k 有交集，讓 j 跟著 k 去同一座設施
   - 否則在 F_j 中開一座最便宜的設施
5) 輸出 rounding 後成本，對照 LP 成本與理論上界
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

# --- Optional dependencies (集中到檔案最上面) ---
try:
    import numpy as np  # 用於產生隨機測資（可無）
except Exception:
    np = None  # type: ignore

try:
    import pulp  # 用於解 LP 鬆弛（可無）
except Exception:
    pulp = None  # type: ignore


# -------------------------------
# 問題資料結構
# -------------------------------

@dataclass
class UFLInstance:
    """UFL 問題資料：f[i] 開設成本，d[i][j] 距離（建議為 metric）。"""
    f: List[float]                # len = |F|
    d: List[List[float]]          # shape = (|F|, |D|)

    @property
    def n_facilities(self) -> int:
        return len(self.f)

    @property
    def n_clients(self) -> int:
        return len(self.d[0]) if self.d else 0


# -------------------------------
# LP 解（分數解）容器
# -------------------------------

@dataclass
class LPSolution:
    """線性鬆弛的分數解：y[i] ∈ [0,1]；x[i][j] ∈ [0,1] 且 Σ_i x[i][j] = 1。"""
    y: List[float]                     # len = |F|
    x: List[List[float]]               # shape = (|F|, |D|)

    def cost_facility(self, inst: UFLInstance) -> float:
        return sum(inst.f[i] * self.y[i] for i in range(inst.n_facilities))

    def cost_distance(self, inst: UFLInstance) -> float:
        m, n = inst.n_facilities, inst.n_clients
        return sum(inst.d[i][j] * self.x[i][j] for i in range(m) for j in range(n))


# -------------------------------
# UFL LP 求解（可選：需要 PuLP）
# -------------------------------

def solve_lp_relaxation(inst: UFLInstance) -> Optional[LPSolution]:
    """
    用 PuLP 求 LP 鬆弛解。若未安裝 PuLP，回傳 None 讓呼叫端 fallback。
    Min: Σ_i f_i y_i + Σ_{i,j} d_ij x_ij
    s.t. Σ_i x_ij = 1; x_ij <= y_i; x_ij, y_i >= 0
    """
    if pulp is None:
        return None

    m, n = inst.n_facilities, inst.n_clients
    prob = pulp.LpProblem("UFL_LP_Relaxation", pulp.LpMinimize)

    y = [pulp.LpVariable(f"y_{i}", lowBound=0.0, upBound=1.0) for i in range(m)]
    x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0.0, upBound=1.0)
          for j in range(n)] for i in range(m)]

    # 目標函數
    prob += pulp.lpSum(inst.f[i] * y[i] for i in range(m)) + \
            pulp.lpSum(inst.d[i][j] * x[i][j] for i in range(m) for j in range(n))

    # 約束
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(m)) == 1.0, f"assign_once_{j}"
    for i in range(m):
        for j in range(n):
            prob += x[i][j] <= y[i], f"open_before_assign_{i}_{j}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    y_sol = [v.value() or 0.0 for v in y]
    x_sol = [[var.value() or 0.0 for var in row] for row in x]
    return LPSolution(y=y_sol, x=x_sol)


# -------------------------------
# Rounding 主程序
# -------------------------------

@dataclass
class RoundingResult:
    opened_facilities: Set[int]          # 開設的設施索引
    assignment: List[int]                # assignment[j] = 服務客戶 j 的設施 i
    lp_cost_facility: float
    lp_cost_distance: float
    round_cost_facility: float
    round_cost_distance: float
    c: float
    r: float

def lp_rounding_for_ufl(inst: UFLInstance,
                        c: float = 4/3,
                        lp_solution: Optional[LPSolution] = None,
                        eps: float = 1e-12) -> RoundingResult:
    """
    UFL 的 LP rounding 演算法（完整對應教材步驟）。
    參數：
      - c: 常數 > 1（理論最佳 c=4/3，可得 4-approx）
      - lp_solution: 若已先求得 (x,y) 可直接帶入；否則嘗試用 PuLP 求
    """
    # 0) 取得 LP 分數解；若沒有 PuLP，採用簡易備援分數解
    if lp_solution is None:
        lp_solution = solve_lp_relaxation(inst)

    if lp_solution is None:
        m, n = inst.n_facilities, inst.n_clients
        x = [[0.0 for _ in range(n)] for _ in range(m)]
        for j in range(n):
            idx = sorted(range(m), key=lambda i: inst.d[i][j])[:2]
            x[idx[0]][j] = 0.6
            x[idx[1]][j] = 0.4
        y = [max(x[i][j] for j in range(n)) for i in range(m)]
        lp_solution = LPSolution(y=y, x=x)

    m, n = inst.n_facilities, inst.n_clients

    # 1) LP 成本（對照用）
    lp_fac_cost = lp_solution.cost_facility(inst)
    lp_dis_cost = lp_solution.cost_distance(inst)

    # 2) 計算 D_j 與 F_j
    D = [0.0 for _ in range(n)]
    F_sets: List[Set[int]] = [set() for _ in range(n)]
    for j in range(n):
        exp_dist = sum(inst.d[i][j] * lp_solution.x[i][j] for i in range(m))
        D[j] = c * exp_dist
        for i in range(m):
            if lp_solution.x[i][j] > eps and inst.d[i][j] <= D[j] + 1e-15:
                F_sets[j].add(i)
        if not F_sets[j]:
            support = [i for i in range(m) if lp_solution.x[i][j] > eps]
            if support:
                i_star = min(support, key=lambda i: inst.d[i][j])
            else:
                i_star = min(range(m), key=lambda i: inst.d[i][j])
            F_sets[j].add(i_star)

    # 3) 依 D_j 非遞減排序處理客戶
    order = sorted(range(n), key=lambda j: D[j])

    # 4) 指派與開設
    assignment = [-1] * n
    opened: Set[int] = set()
    clients_on_facility: Dict[int, Set[int]] = defaultdict(set)
    processed_clients: Set[int] = set()
    r = 1.0 - (1.0 / c)

    for j in order:
        follow_i = None
        if processed_clients:
            for k in processed_clients:
                i_k = assignment[k]
                if i_k != -1 and i_k in F_sets[j]:
                    follow_i = i_k
                    break

        if follow_i is not None:
            assignment[j] = follow_i
            clients_on_facility[follow_i].add(j)
        else:
            i_star = min(F_sets[j], key=lambda i: inst.f[i])
            opened.add(i_star)
            assignment[j] = i_star
            clients_on_facility[i_star].add(j)

        processed_clients.add(j)

    # 5) rounding 後成本
    round_fac_cost = sum(inst.f[i] for i in opened)
    round_dis_cost = sum(inst.d[assignment[j]][j] for j in range(n))

    return RoundingResult(
        opened_facilities=opened,
        assignment=assignment,
        lp_cost_facility=lp_fac_cost,
        lp_cost_distance=lp_dis_cost,
        round_cost_facility=round_fac_cost,
        round_cost_distance=round_dis_cost,
        c=c,
        r=r
    )


# -------------------------------
#（可選）展示用：小型測試
# -------------------------------

if __name__ == "__main__":
    # 例子：5 個設施、8 個客戶；距離可用隨機 metric（簡化示範）
    if np is not None:
        rng = np.random.default_rng(42)
        m, n = 5, 8
        f = (rng.uniform(3.0, 8.0, size=m)).round(2).tolist()
        d = (rng.uniform(1.0, 12.0, size=(m, n))).round(2).tolist()
    else:
        f = [5.0, 6.5, 4.2, 7.3, 5.8]
        d = [
            [3, 7, 4, 6, 8, 9, 5, 6],
            [5, 6, 7, 9, 4, 3, 8, 7],
            [2, 8, 5, 7, 6, 6, 4, 5],
            [7, 9, 6, 3, 8, 7, 9, 6],
            [4, 5, 6, 8, 6, 5, 7, 3]
        ]

    inst = UFLInstance(f=f, d=d)
    c = 4/3  # 建議值

    res = lp_rounding_for_ufl(inst, c=c, lp_solution=None)

    print("=== UFL LP Rounding Report ===")
    print(f"|F|={inst.n_facilities}, |D|={inst.n_clients}, c={res.c:.4f}, r=1-1/c={res.r:.4f}")
    print(f"LP  Facility  Cost = {res.lp_cost_facility:.3f}")
    print(f"LP  Distance  Cost = {res.lp_cost_distance:.3f}")
    print(f"RND Facility  Cost = {res.round_cost_facility:.3f}  (theory ≤ (c/(c-1))*LP_fac)")
    print(f"RND Distance  Cost = {res.round_cost_distance:.3f} (theory ≤ (3c)*LP_dis)")
    print(f"Opened facilities = {sorted(list(res.opened_facilities))}")
    print(f"Assignment (client j -> facility i) = {res.assignment}")
