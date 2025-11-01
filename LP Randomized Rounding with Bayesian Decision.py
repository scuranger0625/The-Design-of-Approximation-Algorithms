# -*- coding: utf-8 -*-
"""
Bayesian-style LP Randomized Rounding Demo (MAX-SAT, Section 5.3: Flipping Biased Coins)
----------------------------------------------------------------------------------------
說明：
 本程式示範「有偏硬幣（biased coin）」的隨機化捨入法，
 對 MAX-SAT 問題做近似解，並使用「條件期望法」做確定化版本。
 對應到貝葉斯決策論：信念分佈 → 行動機率 → 最大化期望效用。
"""
import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd

# -----------------------------
# 基本參數
# -----------------------------
N_VARS = 10
M_CLAUSES = 30
PROB_NEG = 0.5
UNIT_CLAUSE_RATE = 0.25
WEIGHT_RANGE = (1, 10)
SEED = 42
GOLDEN_P = (math.sqrt(5) - 1) / 2.0  # ≈0.618
NUM_SAMPLES = 1000

# -----------------------------
# 資料結構
# -----------------------------
@dataclass
class Clause:
    literals: Tuple[int, ...]
    weight: float

@dataclass
class MaxSATInstance:
    n_vars: int
    clauses: List[Clause]

# -----------------------------
# 隨機產生 MAX-SAT 實例
# -----------------------------
def generate_random_instance(n_vars, m_clauses,
                             prob_neg=0.5, unit_rate=0.25,
                             w_range=(1,10), seed=None):
    if seed is not None:
        random.seed(seed)
    clauses = []
    for _ in range(m_clauses):
        if random.random() < unit_rate:
            var = random.randint(1, n_vars)
            lit = -var if random.random() < prob_neg else var
            w = random.randint(*w_range)
            clauses.append(Clause((lit,), w))
        else:
            k = random.randint(2, 5)
            lits = []
            idxs = random.sample(range(1, n_vars + 1), k)
            for var in idxs:
                lit = -var if random.random() < prob_neg else var
                lits.append(lit)
            w = random.randint(*w_range)
            clauses.append(Clause(tuple(lits), w))
    return MaxSATInstance(n_vars, clauses)

# -----------------------------
# 子句滿足機率
# -----------------------------
def clause_satisfied_prob_given_p(clause, p):
    a = sum(1 for lit in clause.literals if lit < 0)
    b = sum(1 for lit in clause.literals if lit > 0)
    return 1.0 - (p ** a) * ((1 - p) ** b)

# -----------------------------
# 評分：具體指派
# -----------------------------
def eval_assignment(instance, assign):
    total = 0.0
    for clause in instance.clauses:
        sat = False
        for lit in clause.literals:
            var = abs(lit)
            val = assign.get(var, False)
            if (lit > 0 and val) or (lit < 0 and not val):
                sat = True
                break
        if sat:
            total += clause.weight
    return total

# -----------------------------
# 隨機化捨入
# -----------------------------
def randomized_rounding(instance, p, seed=None):
    if seed is not None:
        random.seed(seed)
    return {i: random.random() < p for i in range(1, instance.n_vars + 1)}

# -----------------------------
# 理論下界
# -----------------------------
def theoretical_lower_bounds(instance, p):
    sum_w = sum(c.weight for c in instance.clauses)
    v = [0.0] * (instance.n_vars + 1)
    has_neg_unit = False
    for c in instance.clauses:
        if len(c.literals) == 1 and c.literals[0] < 0:
            has_neg_unit = True
            var = -c.literals[0]
            v[var] = max(v[var], c.weight)
    sum_vi = sum(v[1:])
    lb_no_neg_units = min(p, 1 - p ** 2) * sum_w
    lb_general = p * (sum_w - sum_vi)
    return dict(sum_w=sum_w, sum_vi=sum_vi, has_neg_unit=has_neg_unit,
                LB_no_neg_units=lb_no_neg_units, LB_general=lb_general)

# -----------------------------
# 條件期望法（去隨機化）
# -----------------------------
def expected_clause_satisfied_given_partial(clause, partial, p):
    undecided_pos, undecided_neg = 0, 0
    for lit in clause.literals:
        var = abs(lit)
        val = partial.get(var, None)
        if val is None:
            if lit > 0: undecided_pos += 1
            else: undecided_neg += 1
        else:
            if (lit > 0 and val) or (lit < 0 and not val):
                return 1.0
    fail_prob = (p ** undecided_neg) * ((1 - p) ** undecided_pos)
    return 1.0 - fail_prob

def conditional_expectation_deterministic(instance, p):
    partial = {i: None for i in range(1, instance.n_vars + 1)}
    def total_exp_weight(partial_assign):
        exp_total = 0.0
        for c in instance.clauses:
            exp_total += c.weight * expected_clause_satisfied_given_partial(c, partial_assign, p)
        return exp_total
    for i in range(1, instance.n_vars + 1):
        partial[i] = True; ew_t = total_exp_weight(partial)
        partial[i] = False; ew_f = total_exp_weight(partial)
        partial[i] = (ew_t >= ew_f)
    return {i: bool(partial[i]) for i in partial}

# -----------------------------
# 主流程
# -----------------------------
def run_demo():
    inst = generate_random_instance(N_VARS, M_CLAUSES, PROB_NEG, UNIT_CLAUSE_RATE, WEIGHT_RANGE, SEED)
    p = GOLDEN_P
    bounds = theoretical_lower_bounds(inst, p)

    # Monte Carlo 估計
    random_vals = []
    rng = random.Random(SEED)
    for _ in range(NUM_SAMPLES):
        assign = randomized_rounding(inst, p, seed=rng.randint(0, 1e9))
        val = eval_assignment(inst, assign)
        random_vals.append(val)
    mc_mean = sum(random_vals) / len(random_vals)
    mc_std = (sum((x - mc_mean)**2 for x in random_vals) / len(random_vals))**0.5

    det_assign = conditional_expectation_deterministic(inst, p)
    det_val = eval_assignment(inst, det_assign)

    print("===== MAX-SAT 偏幣 LP 隨機化（含去隨機化）比較 =====")
    print(f"Σw_j = {bounds['sum_w']:.2f}, Σv_i = {bounds['sum_vi']:.2f}, 含否定單元? {bounds['has_neg_unit']}")
    print(f"理論下界（無否定）= {bounds['LB_no_neg_units']:.2f}")
    print(f"理論下界（一般）  = {bounds['LB_general']:.2f}")
    print(f"偏幣隨機法 Monte Carlo 平均 = {mc_mean:.2f} ± {mc_std:.2f}")
    print(f"條件期望法確定化結果 = {det_val:.2f}")
    print("\n前10個子句的滿足機率（p=0.618）:")
    for j, c in enumerate(inst.clauses[:10], start=1):
        prob = clause_satisfied_prob_given_p(c, p)
        print(f"  C{j}: {c.literals}, w={c.weight}, P(satisfied)={prob:.3f}")

# 執行
run_demo()
