# -*- coding: utf-8 -*-
"""
UFL (Uncapacitated Facility Location) — LP Rounding Demo
-------------------------------------------------------
這支程式示範整條流程：
1)（可選）用 LP 鬆弛求出分數解 (x_ij, y_i)
2) 設定常數 c (>1)，計算每個客戶的門檻距離 D_j = c * sum_i d_ij x_ij
3) 定義候選集合 F_j = { i | x_ij > 0 且 d_ij <= D_j }
4) 依 D_j 非遞減排序處理客戶（Challenge 2 的關鍵步驟）
   - 若 F_j 與已連線客戶 k 的 F_k 有交集，讓 j 跟著 k 去同一座設施（共享，避免多開）
   - 否則在 F_j 中開一座最便宜的設施（成本 ≤ (1/r) * Σ_{i∈F_j} f_i y_i）
5) 計算 rounding 後的設施成本與距離成本，對照 LP 成本與理論上界
   - Cost_facility ≤ (c/(c-1)) * OPT_facility^LP
   - Cost_distance ≤ 3c * OPT_distance^LP
   - 總近似比 ≤ max{3c, c/(c-1)}

寫法偏教學導向（可讀性 > 微效能），方便你逐行比對教材。
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

try:
    import numpy as np
except Exception:
    np = None  # 若沒有 numpy 也能跑，只是少了小工具


# -------------------------------
# 問題資料結構
# -------------------------------

@dataclass
class UFLInstance:
    """
    UFL 問題的基本資料：
    - f[i]: 設施 i 的開設成本 f_i
    - d[i][j]: 設施 i 到客戶 j 的距離 d_ij（必須滿足三角不等式才有 Challenge 2 的保證）
    """
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
    """
    線性鬆弛的分數解：
    - y[i] ∈ [0,1]
    - x[i][j] ∈ [0,1] 且 Σ_i x[i][j] = 1
    """
    y: List[float]                     # len = |F|
    x: List[List[float]]               # shape = (|F|, |D|)

    def cost_facility(self, inst: UFLInstance) -> float:
        return sum(inst.f[i] * self.y[i] for i in range(inst.n_facilities))

    def cost_distance(self, inst: UFLInstance) -> float:
        m, n = inst.n_facilities, inst.n_clients
        total = 0.0
        for i in range(m):
            for j in range(n):
                total += inst.d[i][j] * self.x[i][j]
        return total


# -------------------------------
# UFL LP 求解（可選：需要 PuLP）
# -------------------------------

def solve_lp_relaxation(inst: UFLInstance) -> Optional[LPSolution]:
    """
    嘗試用 PuLP 求 LP 鬆弛解。若環境沒有 PuLP，回傳 None。
    Minimize: Σ_i f_i y_i + Σ_{i,j} d_ij x_ij
    s.t.     Σ_i x_ij = 1, ∀j
             x_ij <= y_i, ∀i,j
             x_ij, y_i >= 0
    """
    try:
        import pulp
    except Exception:
        return None  # 沒有 PuLP，呼叫端會 fallback

    m, n = inst.n_facilities, inst.n_clients

    # 建立 LP
    prob = pulp.LpProblem("UFL_LP_Relaxation", pulp.LpMinimize)

    # 變數：y_i ∈ [0,1]、x_ij ∈ [0,1]
    y = [pulp.LpVariable(f"y_{i}", lowBound=0.0, upBound=1.0) for i in range(m)]
    x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0.0, upBound=1.0)
          for j in range(n)] for i in range(m)]

    # 目標函數
    prob += pulp.lpSum(inst.f[i] * y[i] for i in range(m)) + \
            pulp.lpSum(inst.d[i][j] * x[i][j] for i in range(m) for j in range(n))

    # 約束：每個客戶 j，Σ_i x_ij = 1
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(m)) == 1.0, f"assign_once_{j}"

    # 約束：x_ij <= y_i
    for i in range(m):
        for j in range(n):
            prob += x[i][j] <= y[i], f"open_before_assign_{i}_{j}"

    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    # 若解不可行或未最佳，這裡仍會取變數值當作一個可行分數解嘗試（教學示範）
    y_sol = [v.value() if v.value() is not None else 0.0 for v in y]
    x_sol = [[var.value() if var.value() is not None else 0.0 for var in row] for row in x]

    return LPSolution(y=y_sol, x=x_sol)


# -------------------------------
# Rounding 主程序
# -------------------------------

@dataclass
class RoundingResult:
    opened_facilities: Set[int]          # 最後開了哪些設施
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
      - inst: 問題資料
      - c: 常數 > 1。理論最佳值為 4/3，可得到 4-approx。
      - lp_solution: 若你已經有 (x,y) 的 LP 分數解，可直接傳入；否則會嘗試用 PuLP 求。
      - eps: 判斷 x_ij > 0 的容忍閾值（避免浮點數雜訊）

    回傳：RoundingResult（含所有成本、開設集合與指派）
    """

    # 0) 取得 LP 分數解；若沒有 PuLP，嘗試給一個「分數啟發式」
    if lp_solution is None:
        lp_solution = solve_lp_relaxation(inst)

    if lp_solution is None:
        # 教學用簡易備援：把每個客戶 j 的 x_ij 分數分在（前兩名最便宜距離）的設施上
        # 這樣 Σ_i x_ij = 1 且 x_ij ∈ [0,1]；y_i 用 max_j x_ij 當作啟發式
        m, n = inst.n_facilities, inst.n_clients
        x = [[0.0 for _ in range(n)] for _ in range(m)]
        for j in range(n):
            # 取距離最小的兩個設施當「支持集」
            idx = sorted(range(m), key=lambda i: inst.d[i][j])[:2]
            x[idx[0]][j] = 0.6
            x[idx[1]][j] = 0.4
        y = [max(x[i][j] for j in range(n)) for i in range(m)]
        lp_solution = LPSolution(y=y, x=x)

    m, n = inst.n_facilities, inst.n_clients

    # 1) 計算 LP 成本（供對照）
    lp_fac_cost = lp_solution.cost_facility(inst)
    lp_dis_cost = lp_solution.cost_distance(inst)

    # 2) 計算 D_j 與 F_j
    #    D_j = c * Σ_i d_ij x_ij
    #    F_j = {i | x_ij > 0 且 d_ij <= D_j}
    D = [0.0 for _ in range(n)]
    F_sets: List[Set[int]] = [set() for _ in range(n)]
    for j in range(n):
        exp_dist = sum(inst.d[i][j] * lp_solution.x[i][j] for i in range(m))  # Σ d_ij x_ij
        D[j] = c * exp_dist
        for i in range(m):
            if lp_solution.x[i][j] > eps and inst.d[i][j] <= D[j] + 1e-15:
                F_sets[j].add(i)
        # 保底：理論上 F_j 不會空（因為 D_j >= Σ d_ij x_ij 的加權平均 ≥ min d_ij in support）
        if not F_sets[j]:
            # 若數值誤差導致空集合，就至少把「支撐集中距離最小」的那個放進來
            support = [i for i in range(m) if lp_solution.x[i][j] > eps]
            if support:
                i_star = min(support, key=lambda i: inst.d[i][j])
            else:
                i_star = min(range(m), key=lambda i: inst.d[i][j])
            F_sets[j].add(i_star)

    # 3) 依 D_j 非遞減排序處理客戶（Challenge 2 的關鍵）
    order = sorted(range(n), key=lambda j: D[j])

    # 4) 逐一指派
    assignment = [-1] * n              # assignment[j] = 指派到的設施 i
    opened: Set[int] = set()           # 已開設施
    # 方便找「已連線客戶 k」且 F_j ∩ F_k ≠ ∅
    # 我們用一個 map: facility i -> 已指派到 i 的客戶集合（反向索引）
    clients_on_facility: Dict[int, Set[int]] = defaultdict(set)

    # 方便找「某個已處理客戶 k，其 F_k 與 F_j 有交集」
    processed_clients: Set[int] = set()

    # 這裡也會用到 Challenge 1 的「成本保證」資訊（for 註解/檢查）
    # r = 1 - 1/c（由 Markov 的界），設施成本 ≤ (1/r) * LP_fac
    r = 1.0 - (1.0 / c)

    for j in order:
        # 優先嘗試「跟隨」策略：找一個已處理的 k 使 F_j ∩ F_k ≠ ∅
        follow_i = None
        if processed_clients:
            # 合併所有已開設施（因為跟隨指向的是「已連線客戶的設施」）
            # 我們先找：是否存在 k，使得 assignment[k] ∈ F_j（代表交集非空且 j 可直接共用）
            candidate_i = None
            for k in processed_clients:
                i_k = assignment[k]
                if i_k != -1 and i_k in F_sets[j]:
                    candidate_i = i_k
                    break
            follow_i = candidate_i

        if follow_i is not None:
            # 直接跟著去同一座設施（避免多開、符合 Challenge 2）
            assignment[j] = follow_i
            clients_on_facility[follow_i].add(j)
            # 距離成本在理論上以三角不等式：d(j, follow_i) <= D_j + 2 D_k <= 3 D_j
        else:
            # 若沒有可跟隨的交集，就在 F_j 裡挑一座最便宜的設施開（Challenge 1 的保證）
            # 「最便宜」對應 f = min_{i∈F_j} f_i，且 f ≤ (1/r) Σ_{i∈F_j} f_i y_i
            i_star = min(F_sets[j], key=lambda i: inst.f[i])
            opened.add(i_star)
            assignment[j] = i_star
            clients_on_facility[i_star].add(j)

        processed_clients.add(j)

    # 5) 計算 rounding 後成本
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
#（可選）展示用：小型隨機測試
# -------------------------------

if __name__ == "__main__":
    # 你可以把這段換成自己的資料；這裡做一個小例子方便直接執行。
    # 例子：5 個設施、8 個客戶；距離用隨機度量（這裡簡化為任意非負；正式分析需 metric）
    rng = None
    if np is not None:
        rng = np.random.default_rng(42)
        m, n = 5, 8
        f = (rng.uniform(3.0, 8.0, size=m)).round(2).tolist()
        # 產生距離矩陣（隨機）；課堂理論假設 metric（含三角不等式），
        # 真要嚴謹可以先生成點座標，再用歐氏距離；這裡從簡示範。
        d = (rng.uniform(1.0, 12.0, size=(m, n))).round(2).tolist()
    else:
        # 沒有 numpy 就放固定數
        f = [5.0, 6.5, 4.2, 7.3, 5.8]
        d = [
            [3, 7, 4, 6, 8, 9, 5, 6],
            [5, 6, 7, 9, 4, 3, 8, 7],
            [2, 8, 5, 7, 6, 6, 4, 5],
            [7, 9, 6, 3, 8, 7, 9, 6],
            [4, 5, 6, 8, 6, 5, 7, 3]
        ]

    inst = UFLInstance(f=f, d=d)

    # 推薦 c = 4/3（近似比 4）；你也可以改成 c=2 看 2→(r=1/2) 與距離 3c=6 的 trade-off
    c = 4/3

    res = lp_rounding_for_ufl(inst, c=c, lp_solution=None)

    # 報告
    print("=== UFL LP Rounding Report ===")
    print(f"|F|={inst.n_facilities}, |D|={inst.n_clients}, c={res.c:.4f}, r=1-1/c={res.r:.4f}")
    print(f"LP  Facility  Cost = {res.lp_cost_facility:.3f}")
    print(f"LP  Distance  Cost = {res.lp_cost_distance:.3f}")
    print(f"RND Facility  Cost = {res.round_cost_facility:.3f}  (theory ≤ (c/(c-1))*LP_fac)")
    print(f"RND Distance  Cost = {res.round_cost_distance:.3f} (theory ≤ (3c)*LP_dis)")
    print(f"Opened facilities = {sorted(list(res.opened_facilities))}")
    print(f"Assignment (client j -> facility i) = {res.assignment}")
