# -*- coding: utf-8 -*-
"""Q2 主程序 v2：删失构造 -> AFT 达标时间模型 -> 当次概率核验 -> 约束型主方案 + 风险情景 -> 误差影响 -> 稳定性。
主方案：组内平均达标概率 >=0.95 的最早时点（DP 最小化总人·周）；风险最小化作为权重情景敏感性。
运行: python q2_pipeline.py   输入: Q1/04_结果   输出: Q2/04_结果/*.csv + q2_run_summary.json
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats as sps, optimize as opt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q1RES = os.path.join(os.path.dirname(ROOT), "Q1", "04_结果")
OUT = os.path.join(ROOT, "04_结果")
os.makedirs(OUT, exist_ok=True)
SEED = 42
log = {}

for f in ["q1_blood_events.csv", "q1_run_summary.json", "q1_effects_main.csv"]:
    if not os.path.isfile(os.path.join(Q1RES, f)):
        sys.exit(f"缺少 Q1 结果文件: {f}")
ev = pd.read_csv(os.path.join(Q1RES, "q1_blood_events.csv"))
q1sum = json.load(open(os.path.join(Q1RES, "q1_run_summary.json"), encoding="utf-8"))
eff = pd.read_csv(os.path.join(Q1RES, "q1_effects_main.csv"))
S_HAT = float(q1sum["单次测量误差估计s_hat(事件内合并方差)"])
log["误差s_hat"] = S_HAT

# ---------------- 1. 删失构造 ----------------
rows = []
for pid, g in ev.sort_values("t_weeks").groupby("pid"):
    t, h = g["t_weeks"].values, g["y_hit"].values
    first_hit = np.argmax(h) if h.any() else None
    if first_hit is None:
        typ, L, R = "右删失", t[-1], np.inf
    elif first_hit == 0:
        typ, L, R = "左删失", 0.0, t[0]
    else:
        typ, L, R = "区间删失", t[first_hit - 1], t[first_hit]
    rev = bool(first_hit is not None and (h[first_hit:] == 0).any())
    rows.append({"pid": pid, "B0": g["B0"].iloc[0], "censor_type": typ, "L": L, "R": R,
                 "n_events": len(t), "t_first": t[0], "t_last": t[-1], "reversed": rev})
cen = pd.DataFrame(rows)
cen.to_csv(os.path.join(OUT, "q2_censoring.csv"), index=False, encoding="utf-8-sig")
log["删失计数"] = cen["censor_type"].value_counts().to_dict()
log["反转人数"] = int(cen["reversed"].sum())
log["反转比例"] = round(float(cen["reversed"].mean()), 4)
cen["B0c"] = cen["B0"] - cen["B0"].mean()

# ---------------- 2. AFT ----------------
def nll_factory(dist):
    def logF(lt, mu, sig):
        z = (lt - mu) / sig
        return sps.norm.logcdf(z) if dist == "lognormal" else np.log1p(-np.exp(-np.exp(z)))
    def logS(lt, mu, sig):
        z = (lt - mu) / sig
        return sps.norm.logsf(z) if dist == "lognormal" else -np.exp(z)
    def nll(par, data):
        a0, a1, lsig = par
        sig = np.exp(lsig)
        mu = a0 + a1 * data["B0c"].values
        m = data["censor_type"].values
        lL = np.log(np.clip(data["L"].values, 1e-9, None))
        lR = np.log(np.where(np.isinf(data["R"]), 1.0, data["R"].values))
        isL, isI, isR = m == "左删失", m == "区间删失", m == "右删失"
        ll = logF(lR[isL], mu[isL], sig).sum()
        d = np.exp(logF(lR[isI], mu[isI], sig)) - np.exp(logF(lL[isI], mu[isI], sig))
        ll += np.log(np.clip(d, 1e-12, None)).sum()
        ll += logS(lL[isR], mu[isR], sig).sum()
        return -ll
    return nll

def fit_aft(dist, data):
    nll = nll_factory(dist)
    best = None
    for a1_0 in [0.0, 0.02, 0.05]:
        r = opt.minimize(nll, [np.log(14.0), a1_0, np.log(0.35)], args=(data,), method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best

# 锚点 A：5 数据集均值恢复
a0_t, a1_t, sig_t = np.log(15.0), 0.030, 0.40
ests = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    T_sim = np.exp(a0_t + a1_t * cen["B0c"].values + sig_t * rng.normal(size=len(cen)))
    sim = cen[["pid", "B0", "B0c"]].copy()
    sim["censor_type"] = "区间删失"; sim["L"] = 0.0; sim["R"] = 1.0
    for pid, g in ev.sort_values("t_weeks").groupby("pid"):
        i = cen.index[cen["pid"] == pid][0]
        ts = g["t_weeks"].values; Ti = T_sim[cen.index.get_loc(i)]
        hit = ts >= Ti
        if hit.all():
            sim.loc[i, ["censor_type", "L", "R"]] = ["左删失", 0.0, ts[0]]
        elif not hit.any():
            sim.loc[i, ["censor_type", "L", "R"]] = ["右删失", ts[-1], np.inf]
        else:
            j = int(np.argmax(hit))
            sim.loc[i, ["censor_type", "L", "R"]] = ["区间删失", ts[j - 1], ts[j]]
    ests.append(fit_aft("lognormal", sim).x)
ests = np.array(ests).mean(axis=0)
recov = {"a0": abs(np.exp(ests[0]) - 15.0) / 15.0, "a1": abs(ests[1] - a1_t) / a1_t,
         "sig": abs(np.exp(ests[2]) - sig_t) / sig_t}
log["锚点A_AFT仿真恢复相对误差(5数据集均值)"] = {k: round(v, 4) for k, v in recov.items()}
assert all(v < 0.15 for v in recov.values()), f"锚点A失败: {recov}"

def fit_with_se(dist, data):
    nll = nll_factory(dist)
    r = fit_aft(dist, data)
    x = r.x; n = len(x); H = np.zeros((n, n)); eps = 1e-4
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n); ei[i] = eps; ej[j] = eps
            H[i, j] = (nll(x + ei + ej, data) - nll(x + ei - ej, data)
                       - nll(x - ei + ej, data) + nll(x - ei - ej, data)) / (4 * eps ** 2)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return r, se

r_ln, se_ln = fit_with_se("lognormal", cen)
r_wb, se_wb = fit_with_se("weibull", cen)
aic_ln, aic_wb = 2 * r_ln.fun + 6, 2 * r_wb.fun + 6
dist = "lognormal" if aic_ln <= aic_wb else "weibull"
r_best, se_best = (r_ln, se_ln) if dist == "lognormal" else (r_wb, se_wb)
a0, a1, sig = r_best.x[0], r_best.x[1], np.exp(r_best.x[2])
aft_rows = []
for name, r, se, aic in [("lognormal", r_ln, se_ln, aic_ln), ("weibull", r_wb, se_wb, aic_wb)]:
    for k, lab in enumerate(["alpha0", "alpha1_B0c", "log_sigma"]):
        v = r.x[k]
        aft_rows.append({"分布": name, "项": lab, "估计": v, "SE": se[k],
                         "CI下": v - 1.96 * se[k], "CI上": v + 1.96 * se[k], "AIC": aic, "采用": name == dist})
pd.DataFrame(aft_rows).to_csv(os.path.join(OUT, "q2_aft_fit.csv"), index=False, encoding="utf-8-sig")
z_a1 = r_best.x[1] / se_best[1]
log["AFT采用分布"] = dist
log["AFT参数"] = {"alpha0": round(float(a0), 4), "alpha1_B0c": round(float(a1), 5),
                  "sigma": round(float(sig), 4), "alpha1_p": float(2 * sps.norm.sf(abs(z_a1)))}
log["达标时间中位数_BMI均值处(周)"] = round(float(np.exp(a0)), 2)
log["BMI每加1达标中位时间倍数"] = round(float(np.exp(a1)), 4)
log["锚点B_alpha1方向(BMI高达标晚)"] = bool(a1 > 0)

TGRID = np.arange(11.0, 25.01, 0.5)
def F_aft(t, b0c_vals):
    t = np.atleast_1d(t)[:, None]
    z = (np.log(t) - a0 - a1 * np.atleast_1d(b0c_vals)[None, :]) / sig
    return sps.norm.cdf(z) if dist == "lognormal" else 1 - np.exp(-np.exp(z))
F_grid = F_aft(TGRID, cen["B0c"].values)

# ---------------- 3. 当次达标概率 GEE：持续达标假设核验 ----------------
gev = ev.copy()
gev["tc"] = gev["t_weeks"] - 18; gev["tc2"] = gev["tc"] ** 2
gev["B0c"] = gev["B0"] - cen["B0"].mean()
g_lin = smf.gee("y_hit ~ tc + B0c", groups="pid", data=gev, family=sm.families.Binomial()).fit()
g_qd = smf.gee("y_hit ~ tc + tc2 + B0c", groups="pid", data=gev, family=sm.families.Binomial()).fit()
use_qd = bool(g_qd.pvalues["tc2"] < 0.05)
gmod = g_qd if use_qd else g_lin
log["当次模型二次项保留"] = use_qd
def p_gee(t, b0c_vals):
    t = np.atleast_1d(t)[:, None]; b = np.atleast_1d(b0c_vals)[None, :]
    eta = gmod.params["Intercept"] + gmod.params["tc"] * t + gmod.params["B0c"] * b
    if use_qd:
        eta = eta + gmod.params["tc2"] * t ** 2
    return 1 / (1 + np.exp(-eta))
P_grid = p_gee(TGRID, cen["B0c"].values)
mad = float(np.mean(np.abs(F_grid - P_grid))); maxd = float(np.max(np.abs(F_grid - P_grid)))
log["F与当次p比较"] = {"平均绝对差": round(mad, 4), "最大绝对差": round(maxd, 4),
                        "结论": ("F 为删失模型，回答“首次达标时间”；当次 p 受反转影响。高 BMI 尾端差异"
                                f"{maxd:.2f}，推荐时点以 F 为主口径、当次 p 为对照，两口径均报告")}

# ---------------- 4. 主方案：可靠度约束 + DP 最小化总人·周 ----------------
order = cen.sort_values("B0")
oidx = order.index.values  # cen 为 RangeIndex，此即按 BMI 排序后的原位置
B0v = order["B0"].values
P_ord = F_grid[:, oidx]
P_gee_ord = P_grid[:, oidx]
N = len(order)
MIN_N = 30
P_THRESH = 0.95

def earliest_t(pbar, thresh=P_THRESH):
    ok = np.where(pbar >= thresh)[0]
    if len(ok):
        return int(ok[0]), True
    return int(np.argmax(pbar)), False

def optimize_constraint(P, gmax=5, min_n=MIN_N):
    """每段费用 = n_seg * 最早可行时点索引（不可行时取 p 最大并标记）。"""
    n = P.shape[1]
    Psum = np.hstack([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)])
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def seg(i, j):
        m = j - i
        pbar = (Psum[:, j] - Psum[:, i]) / m
        k, feas = earliest_t(pbar)
        return m * k, k, feas
    results = {}
    for g in range(2, gmax + 1):
        INF = 1e18
        dp = np.full((g + 1, n + 1), INF); pre = np.full((g + 1, n + 1), -1)
        dp[0, 0] = 0.0
        for k in range(1, g + 1):
            for j in range(min_n * k, n - min_n * (g - k) + 1):
                for i in range(min_n * (k - 1), j - min_n + 1):
                    if dp[k - 1, i] >= INF:
                        continue
                    c = dp[k - 1, i] + seg(i, j)[0]
                    if c < dp[k, j]:
                        dp[k, j] = c; pre[k, j] = i
        cuts = []; j = n
        for k in range(g, 0, -1):
            i = pre[k, j]; cuts.append((i, j)); j = i
        results[g] = {"cost": dp[g, n], "cuts": cuts[::-1]}
    return results

base = optimize_constraint(P_ord)
mean_weeks = {g: base[g]["cost"] / N * 0.5 + 11 for g in base}
log["约束方案各组数平均推荐孕周"] = {g: round(v, 3) for g, v in mean_weeks.items()}
g_chosen = 2
for g in range(3, 6):
    if mean_weeks[g - 1] - mean_weeks[g] >= 0.25:
        g_chosen = g
    else:
        break
log["肘部规则(再分组改善<0.25周停止)选组数"] = g_chosen

def bounds_of(cuts):
    out = []
    for (i, j) in cuts[:-1]:
        out.append((B0v[j - 1] + B0v[j]) / 2)
    return out

def group_rows(cuts, P, label, thresh=P_THRESH):
    Psum = np.hstack([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)])
    out = []
    for gi, (i, j) in enumerate(cuts, 1):
        m = j - i
        pbar = (Psum[:, j] - Psum[:, i]) / m
        k, feas = earliest_t(pbar, thresh)
        blo = 20.0 if i == 0 else (B0v[i - 1] + B0v[i]) / 2
        bhi = np.inf if j == N else (B0v[j - 1] + B0v[j]) / 2
        out.append({"方案": label, "组": gi,
                    "BMI区间": f"[{blo:.1f}, {bhi:.1f})" if np.isfinite(bhi) else f"[{blo:.1f}, ∞)",
                    "组内BMI范围": f"{B0v[i]:.2f}~{B0v[j-1]:.2f}", "人数": m,
                    "推荐孕周": float(TGRID[k]), "p(推荐时点)": round(float(pbar[k]), 4),
                    "可靠度约束满足": feas})
    return out

cuts_main = base[g_chosen]["cuts"]
main_df = pd.DataFrame(group_rows(cuts_main, P_ord, "约束型主方案(F口径,p≥0.95最早)"))
# 当次 p 口径对照（同样边界）
gee_rows = group_rows(cuts_main, P_gee_ord, "约束型对照(当次p口径)")
main_df = pd.concat([main_df, pd.DataFrame(gee_rows)], ignore_index=True)
main_df.to_csv(os.path.join(OUT, "q2_groups_main.csv"), index=False, encoding="utf-8-sig")
log["主方案"] = main_df[main_df["方案"].str.contains("主方案")].to_dict("records")
log["主方案分界点"] = [round(x, 2) for x in bounds_of(cuts_main)]

# ---------------- 5. 风险最小化情景（同边界） ----------------
def D_delay(t, t0=12.0):
    return np.clip((t - t0) / (27 - t0), 0, 1)
risk_rows = []
for wf, wd in [(1, 1), (2, 1), (1, 2), (4, 1), (1, 4)]:
    for gi, (i, j) in enumerate(cuts_main, 1):
        pbar = P_ord[:, i:j].mean(axis=1)
        R = wf * (1 - pbar) + wd * D_delay(TGRID)
        k = int(np.argmin(R))
        risk_rows.append({"情景": f"wf={wf},wd={wd}", "组": gi, "推荐孕周": float(TGRID[k]),
                          "p(t*)": round(float(pbar[k]), 4), "综合风险": round(float(R[k]), 4)})
risk_df = pd.DataFrame(risk_rows)
risk_df.to_csv(os.path.join(OUT, "q2_risk_scenarios.csv"), index=False, encoding="utf-8-sig")
trange = risk_df.groupby("组")["推荐孕周"].agg(["min", "max"])
log["风险情景t*范围(周)"] = {int(g): [float(trange.loc[g, "min"]), float(trange.loc[g, "max"])] for g in trange.index}

# 锚点 C：极端权重（DP 风险版）
def optimize_risk(P, wf, wd, g=2, min_n=MIN_N):
    n = P.shape[1]
    Psum = np.hstack([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)])
    Dv = D_delay(TGRID)
    def cost(i, j):
        m = j - i
        pbar = (Psum[:, j] - Psum[:, i]) / m
        R = wf * (1 - pbar) + wd * Dv
        k = int(np.argmin(R))
        return m * R[k], k
    INF = 1e18
    dp = np.full((g + 1, n + 1), INF); pre = np.full((g + 1, n + 1), -1)
    dp[0, 0] = 0.0
    for k in range(1, g + 1):
        for j in range(min_n * k, n - min_n * (g - k) + 1):
            for i in range(min_n * (k - 1), j - min_n + 1):
                if dp[k - 1, i] >= INF:
                    continue
                c = dp[k - 1, i] + cost(i, j)[0]
                if c < dp[k, j]:
                    dp[k, j] = c; pre[k, j] = i
    cuts = []; j = n
    for k in range(g, 0, -1):
        i = pre[k, j]; cuts.append((i, j)); j = i
    return [cost(i, j)[1] for (i, j) in cuts[::-1]]
t_idx_wd0 = optimize_risk(P_ord, 1.0, 0.0)
t_idx_wf0 = optimize_risk(P_ord, 0.0, 1.0)
log["锚点C_极端权重"] = {"wd=0取候选域末端": all(TGRID[k] == TGRID[-1] for k in t_idx_wd0),
                         "wf=0取最早时点": all(TGRID[k] == TGRID[0] for k in t_idx_wf0)}
assert log["锚点C_极端权重"]["wd=0取候选域末端"] and log["锚点C_极端权重"]["wf=0取最早时点"]

# ---------------- 6. 误差影响 ----------------
def q1coef(term):
    return float(eff[(eff["模型"] == "M0") & (eff["项"] == term)]["估计"].iloc[0])
b0_, b1_, b2_, b4_ = q1coef("Intercept"), q1coef("t_c"), q1coef("B0_c"), q1coef("t_c:B0_c")
gam_ = q1coef("log_reads")
s2 = q1coef("sigma_b^2(随机截距方差)") + q1coef("sigma^2(残差方差)")
lr_med = float(ev["log_reads"].median())
tc_g = TGRID - 18
m_curve = (b0_ + b1_ * tc_g[:, None] + b2_ * cen["B0c"].values[None, :]
           + b4_ * np.outer(tc_g, cen["B0c"].values) + gam_ * lr_med)
p0 = 1 - sps.norm.cdf((0.04 - m_curve) / np.sqrt(s2))
p1 = 1 - sps.norm.cdf((0.04 - m_curve) / np.sqrt(s2 + S_HAT ** 2))
log["三模型一致性"] = {"F_vs_Q1边际_平均绝对差": round(float(np.mean(np.abs(F_grid - p0))), 4),
                        "当次p_vs_Q1边际_平均绝对差": round(float(np.mean(np.abs(P_grid - p0))), 4)}
P0_ord, P1_ord = p0[:, oidx], p1[:, oidx]
err_rows = []
for gi, (i, j) in enumerate(cuts_main, 1):
    for lab, P in [("无测量误差", P0_ord), ("含测量误差ŝ", P1_ord)]:
        pbar = P[:, i:j].mean(axis=1)
        k, feas = earliest_t(pbar)
        err_rows.append({"组": gi, "口径": lab, "推荐孕周": float(TGRID[k]),
                         "p(推荐时点)": round(float(pbar[k]), 4), "约束满足": feas})
    # 当次误判率（推荐时点，主方案 F 口径时点）
    pbarL = P0_ord[:, i:j].mean(axis=1); pbarO = P1_ord[:, i:j].mean(axis=1)
    kL, _ = earliest_t(pbarL)
    flip = float((pbarL[kL] * (1 - pbarO[kL]) + (1 - pbarL[kL]) * pbarO[kL]))
    err_rows.append({"组": gi, "口径": "当次误判率估计(Q1边际模型)", "推荐孕周": float(TGRID[kL]),
                     "p(推荐时点)": round(flip, 4), "约束满足": ""})
err_df = pd.DataFrame(err_rows)
err_df.to_csv(os.path.join(OUT, "q2_error_impact.csv"), index=False, encoding="utf-8-sig")
piv = err_df[err_df["口径"].str.contains("误差") & ~err_df["口径"].str.contains("误判")].pivot(index="组", columns="口径", values="推荐孕周")
log["误差致t*漂移(周)"] = {int(g): float(piv.loc[g, "含测量误差ŝ"] - piv.loc[g, "无测量误差"]) for g in piv.index}

# ---------------- 7. 校准 ----------------
cal = []
cen["bmi3"] = pd.qcut(cen["B0"], 3, labels=["低", "中", "高"])
evcal = ev.copy(); evcal["bmi3"] = evcal["pid"].map(cen.set_index("pid")["bmi3"])
evcal["tbin"] = pd.cut(evcal["t_weeks"], [10, 13, 16, 20, 26])
for (b3, tb), g in evcal.groupby(["bmi3", "tbin"], observed=True):
    mid = tb.mid
    cal.append({"BMI组": b3, "时间箱": str(tb), "事件数": len(g),
                "观测当次达标率": round(g["y_hit"].mean(), 4),
                "预测p(GEE)": round(float(p_gee(mid, (g["B0"] - cen["B0"].mean()).values).mean()), 4),
                "预测F(AFT)": round(float(F_aft(mid, (g["B0"] - cen["B0"].mean()).values).mean()), 4)})
cal_df = pd.DataFrame(cal)
cal_df.to_csv(os.path.join(OUT, "q2_calibration.csv"), index=False, encoding="utf-8-sig")
log["校准平均绝对误差"] = {"p_GEE": round(float((cal_df["观测当次达标率"] - cal_df["预测p(GEE)"]).abs().mean()), 4),
                            "F_AFT": round(float((cal_df["观测当次达标率"] - cal_df["预测F(AFT)"]).abs().mean()), 4)}

# ---------------- 8. bootstrap 稳定性（主方案） ----------------
B_BOOT = 200
stab = []
for b in range(B_BOOT):
    rng_b = np.random.default_rng(SEED + b)
    bc = cen.iloc[rng_b.choice(len(cen), size=len(cen), replace=True)].reset_index(drop=True)
    bc["B0c"] = bc["B0"] - cen["B0"].mean()
    try:
        rb = fit_aft(dist, bc)
        zb = (np.log(TGRID)[:, None] - rb.x[0] - rb.x[1] * bc["B0c"].values[None, :]) / np.exp(rb.x[2])
        Fb = sps.norm.cdf(zb) if dist == "lognormal" else 1 - np.exp(-np.exp(zb))
        ob = bc.sort_values("B0")
        Fb_ord = Fb[:, ob.index.values]  # bc 为 RangeIndex，按 BMI 排序取原位置
        ob0 = ob["B0"].values
        # 用 bc 的顺序做 DP（bounds_of 依赖 B0v，局部重算）
        res = optimize_constraint(Fb_ord, gmax=g_chosen)
        cuts_b = res[g_chosen]["cuts"]
        nb = len(ob0)
        bounds = [round((ob0[j - 1] + ob0[j]) / 2, 1) for (i, j) in cuts_b[:-1]]
        Psum_b = np.hstack([np.zeros((Fb_ord.shape[0], 1)), np.cumsum(Fb_ord, axis=1)])
        for gi, (i, j) in enumerate(cuts_b, 1):
            pbar = (Psum_b[:, j] - Psum_b[:, i]) / (j - i)
            k, feas = earliest_t(pbar)
            stab.append({"b": b, "组": gi, "t*": float(TGRID[k]),
                         "bounds": ";".join(f"{x:.1f}" for x in bounds), "feas": feas})
    except Exception:
        pass
    if (b + 1) % 50 == 0:
        print(f"bootstrap {b + 1}/{B_BOOT}")
stab_df = pd.DataFrame(stab)
stab_df.to_csv(os.path.join(OUT, "q2_stability_boot.csv"), index=False, encoding="utf-8-sig")
tstats = stab_df.groupby("组")["t*"].agg(["mean", "std",
    lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)]).round(3)
tstats.columns = ["t*_均值", "t*_SD", "t*_5%", "t*_95%"]
bcnt = Counter()
for s in stab_df["bounds"]:
    for x in s.split(";"):
        bcnt[x] += 1
with open(os.path.join(OUT, "q2_stability.csv"), "w", encoding="utf-8-sig", newline="") as f:
    tstats.to_csv(f)
    f.write("\n# 分界点频次（前10）\n")
    for x, c in bcnt.most_common(10):
        f.write(f"{x},{c}\n")
log["稳定性_t*"] = tstats.to_dict()
log["稳定性_分界点频次前5"] = bcnt.most_common(5)
if g_chosen == 2:
    bvals = np.array([float(s) for s in stab_df.loc[stab_df["组"] == 1, "bounds"]])
    log["稳定性_分界点90%区间"] = [round(float(np.quantile(bvals, 0.05)), 1),
                                    round(float(np.quantile(bvals, 0.95)), 1)]
log["bootstrap成功次数"] = int(stab_df["b"].nunique())
log["bootstrap约束满足率"] = round(float(stab_df["feas"].mean()), 4)

# 保存曲线供绘图
np.save(os.path.join(OUT, "q2_grid_F.npy"), F_grid[:, oidx])
np.save(os.path.join(OUT, "q2_grid_pGEE.npy"), P_grid[:, oidx])
np.save(os.path.join(OUT, "q2_grid_p0.npy"), P0_ord)
np.save(os.path.join(OUT, "q2_grid_p1.npy"), P1_ord)
order[["pid", "B0"]].to_csv(os.path.join(OUT, "q2_order.csv"), index=False)
json.dump({"cuts": [[int(i), int(j)] for i, j in cuts_main], "g_chosen": int(g_chosen)},
          open(os.path.join(OUT, "q2_cuts.json"), "w"))

with open(os.path.join(OUT, "q2_run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(log, ensure_ascii=False, indent=2, default=str)[:5000])
print("DONE")
