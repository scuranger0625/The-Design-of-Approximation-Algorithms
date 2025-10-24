# -*- coding: utf-8 -*-
"""
Multiway Cut：Isolating Cut Heuristic 的教學範例（k=3）
對應你的投影片 25 頁開始的完整數值例子。

圖 G（無向圖，用無序邊 frozenset 表示）：
    節點：s1, s2, s3（終端）；a（中繼）
    邊權：
        (s1, a) = 2
        (a, s2) = 1
        (a, s3) = 1
        (s2, s3) = 3

終端集合 T = {s1, s2, s3}，k = 3
OPT = {(a,s2),(a,s3),(s2,s3)} 的成本 = 1 + 1 + 3 = 5

我們將：
1) 列出每個終端的 isolating cut 候選（依你簡報的定義）
2) 驗證 sum(cost(C_i)) ≤ 2 * OPT
3) 示範「丟掉最貴的 1 個」後，(k-1) 個割聯集的成本（Case A / Case B）
4) 計算近似比 ALG/OPT，並檢查是否 ≤ 2 - 2/k = 4/3
"""

# === 1) 基本圖與權重設定 ===

# 用 frozenset({u, v}) 當作無向邊的唯一鍵
def e(u, v):
    return frozenset((u, v))

# 權重表
w = {
    e("s1", "a"): 2,
    e("a", "s2"): 1,
    e("a", "s3"): 1,
    e("s2", "s3"): 3,
}

# 幫手：計算一組邊的總成本
def cost(edge_set):
    return sum(w[e] for e in edge_set)

# 幫手：把多個割（邊集合）做聯集，並回傳聯集與其成本
def union_and_cost(cuts):
    U = set()
    for c in cuts:
        U |= c
    return U, cost(U)

# === 2) 依照你的簡報，定義各終端的 isolating cuts ===
# s1 的兩種選法（成本同為 2，但拓撲影響不同）
F1  = { e("s1", "a") }                             # cost = 2
F1p = { e("a", "s2"), e("a", "s3") }               # cost = 1 + 1 = 2

# s2 與 s3 的 isolating cuts（你投影片上的指定集合）
F2 = { e("a", "s2"), e("s2", "s3") }               # cost = 1 + 3 = 4
F3 = { e("a", "s3"), e("s2", "s3") }               # cost = 1 + 3 = 4

# === 3) 最佳解 OPT（你簡報給的目標集合）===
OPT = { e("a", "s2"), e("a", "s3"), e("s2", "s3") }  # cost = 5

# === 4) 檢查 sum(cost(C_i)) ≤ 2 * OPT 的「double counting」事實 ===
# 取 s1 用成本為 2 的最小割（兩個都 2，取 F1 也行）
S = cost(F1) + cost(F2) + cost(F3)   # = 2 + 4 + 4 = 10
OPT_cost = cost(OPT)                 # = 5
lhs = S
rhs = 2 * OPT_cost

print("=== Double Counting 檢查 ===")
print(f"sum_i cost(F_i) = {lhs}")
print(f"2 * OPT = {rhs}")
print(f"是否滿足 sum_i cost(F_i) ≤ 2*OPT ?  {lhs <= rhs}")
print()

# === 5) (k-1) 個割的聯集成本（兩個教學案例） ===
# k = 3，(k-1) = 2，所以要從 {F1 or F1', F2, F3} 中挑兩個做聯集

# Case A：選 F1 與 F2 （對應你簡報的其中一頁）
U_A, ALG_A = union_and_cost([F1, F2])  # 聯集邊：{(s1,a),(a,s2),(s2,s3)}，cost = 2+1+3=6
ratio_A = ALG_A / OPT_cost

# Case B：選 F1' 與 F2（另一頁案例，恰好等於 OPT）
U_B, ALG_B = union_and_cost([F1p, F2]) # 聯集邊：{(a,s2),(a,s3),(s2,s3)}，cost = 1+1+3=5
ratio_B = ALG_B / OPT_cost

print("=== (k-1) 個割的聯集成本與近似比 ===")
print(f"OPT = {sorted([tuple(e) for e in OPT])}, cost(OPT) = {OPT_cost}")
print("--- Case A: 使用 F1 與 F2 ---")
print(f"Union A = {sorted([tuple(e) for e in U_A])}, ALG_A = {ALG_A}, 近似比 ALG_A/OPT = {ratio_A:.3f}")
print("--- Case B: 使用 F1' 與 F2 ---")
print(f"Union B = {sorted([tuple(e) for e in U_B])}, ALG_B = {ALG_B}, 近似比 ALG_B/OPT = {ratio_B:.3f}")
print()

# === 6) 驗證理論上界 (2 - 2/k) ===
k = 3
theory_bound = 2 - 2 / k  # = 4/3 ≈ 1.3333

print("=== (2 - 2/k) 近似上界檢查 ===")
print(f"k = {k}, 2 - 2/k = {theory_bound:.6f}")
print(f"Case A：{ratio_A:.6f} ≤ {theory_bound:.6f} ?  {ratio_A <= theory_bound}")
print(f"Case B：{ratio_B:.6f} ≤ {theory_bound:.6f} ?  {ratio_B <= theory_bound}")
print()

# === 7) 額外：展示「把三個 isolating cuts 全部合起來」與「丟掉最貴的一個」的關係 ===
# 全部合起來（k 個）
U_all, cost_all = union_and_cost([F1, F2, F3])  # 聯集：{(s1,a),(a,s2),(a,s3),(s2,s3)}，cost = 2+1+1+3 = 7
# 丟掉最貴的一個（F2 或 F3 其實都 4，等價），保留 (k-1) 個
# 這裡用「丟掉 F3、保留 F1+F2」來對應 Case A
print("=== 所有 isolating cuts 的聯集 vs. 丟掉最貴一個 ===")
print(f"Union of all 3 cuts = {sorted([tuple(e) for e in U_all])}, cost = {cost_all}")
print(f"Drop the most expensive (cost=4) -> keep F1+F2, cost = {ALG_A} （即 Case A）")
print()
