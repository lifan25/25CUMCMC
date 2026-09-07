# -*- coding: utf-8 -*-
"""Q3 主程序：多因素区间删失 AFT + 协变量筛选 + 约束型 BMI 分组优化 + 误差/分布敏感性 + 稳定性。
运行: python q3_pipeline.py [--xlsx 附件路径]   输入: Q2/04_结果 + Q1/04_结果 + 官方附件   输出: Q3/04_结果/
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats as sps, optimize as opt
from collections import Counter
from functools import lru_cache

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
Q1RES = os.path.join(REPO, "Q1", "04_结果")
Q2RES = os.path.join(REPO, "Q2", "04_结果")
OUT = os.path.join(ROOT, "04_结果")
os.makedirs(OUT, exist_ok=True)
SEED = 42
log = {}

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", default=os.path.join(REPO, "..", "附件.xlsx"))
args = ap.parse_args()
XLSX = os.path.abspath(args.xlsx)
if not os.path.isfile(XLSX):
    sys.exit(f"附件不存在: {XLSX}")

cen = pd.read_csv(os.path.join(Q2RES, "q2_censoring.csv"))
q1sum = json.load(open(os.path.join(Q1RES, "q1_run_summary.json"), encoding="utf-8"))
eff1 = pd.read_csv(os.path.join(Q1RES, "q1_effects_main.csv"))
ev1 = pd.read_csv(os.path.join(Q1RES, "q1_blood_events.csv"))
S_HAT = float(q1sum["单次测量误差估计s_hat(事件内合并方差)"])
log["误差s_hat"] = S_HAT
B0mean = cen["B0"].mean()
cen["B0c"] = cen["B0"] - B0mean

# ---------------- 1. 逐孕妇基线协变量 ----------------
raw = pd.read_excel(XLSX, sheet_name="男胎检测数据")
raw.columns = [str(c).strip() for c in raw.columns]
ev_first = ev1.sort_values("t_weeks").groupby("pid").first().reset_index()[["pid", "age", "height", "IVF"]]
cov = ev_first.copy()
ac = raw.groupby("孕妇代码")["怀孕次数"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
ad = raw.groupby("孕妇代码")["生产次数"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
cov["怀孕次数"] = cov["pid"].map(ac)
cov["生产次数"] = cov["pid"].map(ad)
# 恒定性核查记录
consist = {}
for col in ["年龄", "身高", "IVF妊娠", "怀孕次数", "生产次数"]:
    nun = raw.groupby("孕妇代码")[col].nunique(dropna=True)
    consist[col] = int((nun > 1).sum())
log["协变量非恒定孕妇数(取首次事件值)"] = consist
cen = cen.merge(cov.rename(columns={"age": "age0", "height": "height0"}), on="pid", how="left")
log["协变量缺失人数"] = {c: int(cen[c].isna().sum()) for c in ["age0", "height0", "IVF", "怀孕次数", "生产次数"]}

cen["age_c"] = cen["age0"] - cen["age0"].mean()
cen["height_c"] = cen["height0"] - cen["height0"].mean()
cen["IVF_IUI"] = (cen["IVF"] == "IUI（人工授精）").astype(int)
cen["IVF_IVF"] = (cen["IVF"] == "IVF（试管婴儿）").astype(int)
cen["preg_2"] = (cen["怀孕次数"].astype(str) == "2").astype(int)
cen["preg_3p"] = (cen["怀孕次数"].astype(str) == "≥3").astype(int)
cen["birth_1"] = (cen["生产次数"] == 1).astype(int)
cen["birth_2p"] = (cen["生产次数"] >= 2).astype(int)
log["类别分布"] = {"IVF": cen["IVF"].value_counts().to_dict(),
                    "怀孕次数": cen["怀孕次数"].astype(str).value_counts().to_dict(),
                    "生产次数": cen["生产次数"].astype(str).value_counts().to_dict()}

CANDS = {"age_c": ["age_c"], "height_c": ["height_c"],
         "IVF": ["IVF_IUI", "IVF_IVF"], "preg": ["preg_2", "preg_3p"], "birth": ["birth_1", "birth_2p"]}

# ---------------- 2. 多因素 AFT ----------------
def nll_gen(dist):
    def logF(lt, mu, sig):
        z = (lt - mu) / sig
        return sps.norm.logcdf(z) if dist == "lognormal" else np.log1p(-np.exp(-np.exp(z)))
    def logS(lt, mu, sig):
        z = (lt - mu) / sig
        return sps.norm.logsf(z) if dist == "lognormal" else -np.exp(z)
    def nll(par, data, X):
        a0, a1, lsig = par[0], par[1], par[-1]
        gam = par[2:-1]
        sig = np.exp(lsig)
        mu = a0 + a1 * data["B0c"].values + (X @ gam if X.shape[1] else 0.0)
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

def Xmat(data, cols):
    return data[cols].values.astype(float) if cols else np.zeros((len(data), 0))

def fit_aft(data, cols, dist="lognormal"):
    nll = nll_gen(dist)
    X = Xmat(data, cols)
    k = X.shape[1]
    best = None
    for s0 in [[np.log(14.0), 0.04] + [0.0] * k + [np.log(0.35)],
               [np.log(15.0), 0.0] + [0.0] * k + [np.log(0.5)]]:
        r = opt.minimize(nll, s0, args=(data, X), method="L-BFGS-B")
        if r.success and np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        raise RuntimeError(f"{dist} AFT 未收敛 cols={cols}")
    return best

def fit_se(data, cols, dist="lognormal"):
    nll = nll_gen(dist)
    r = fit_aft(data, cols, dist)
    X = Xmat(data, cols)
    x = r.x; n = len(x); H = np.zeros((n, n)); eps = 1e-4
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n); ei[i] = eps; ej[j] = eps
            H[i, j] = (nll(x + ei + ej, data, X) - nll(x + ei - ej, data, X)
                       - nll(x - ei + ej, data, X) + nll(x - ei - ej, data, X)) / (4 * eps ** 2)
    return r, np.sqrt(np.diag(np.linalg.inv(H)))

# 锚点 A：多变量仿真恢复
a0_t, a1_t, g_t, sig_t = np.log(15.0), 0.030, np.array([0.010, -0.008]), 0.40
colsA = ["age_c", "height_c"]
ests = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    mu_t = a0_t + a1_t * cen["B0c"].values + Xmat(cen, colsA) @ g_t
    T_sim = np.exp(mu_t + sig_t * rng.normal(size=len(cen)))
    sim = cen[["pid", "B0", "B0c"] + colsA].copy()
    sim["censor_type"] = "区间删失"; sim["L"] = 0.0; sim["R"] = 1.0
    for pid, g in ev1.sort_values("t_weeks").groupby("pid"):
        i = sim.index[sim["pid"] == pid][0]
        ts = g["t_weeks"].values; Ti = T_sim[sim.index.get_loc(i)]
        hit = ts >= Ti
        if hit.all():
            sim.loc[i, ["censor_type", "L", "R"]] = ["左删失", 0.0, ts[0]]
        elif not hit.any():
            sim.loc[i, ["censor_type", "L", "R"]] = ["右删失", ts[-1], np.inf]
        else:
            j = int(np.argmax(hit)); sim.loc[i, ["censor_type", "L", "R"]] = ["区间删失", ts[j - 1], ts[j]]
    ests.append(fit_aft(sim, colsA).x)
ests = np.array(ests).mean(axis=0)
truth = np.array([a0_t, a1_t] + g_t.tolist() + [np.log(sig_t)])
recov = np.abs(ests - truth) / np.abs(truth)
abs_err = np.abs(ests - truth)
okA = np.all((recov < 0.15) | (abs_err < 0.005))   # 小真值参数看绝对误差
log["锚点A_多变量仿真恢复(5数据集均值)"] = {"相对误差": [round(float(v), 4) for v in recov],
                                              "绝对误差": [round(float(v), 4) for v in abs_err], "通过": bool(okA)}
assert okA, f"锚点A失败: {recov}"

# 锚点 D：仅 B0c 列时应复现 Q2 冻结参数
r_q2check = fit_aft(cen, [])
q2par = {"a0": 2.0471, "a1": 0.03918, "sig": 0.534}
d_ok = (abs(r_q2check.x[0] - q2par["a0"]) < 2e-3 and abs(r_q2check.x[1] - q2par["a1"]) < 2e-3
        and abs(np.exp(r_q2check.x[-1]) - q2par["sig"]) < 2e-3)
log["锚点D_仅B0c复现Q2参数"] = {"一致": d_ok, "拟合值": [round(float(v), 5) for v in r_q2check.x]}
assert d_ok

# ---------------- 3. 协变量筛选（CV 删失 NLL + 区间不含 0） ----------------
def cv_nll(data, cols, k=5):
    pids = np.array(sorted(data["pid"].unique()))
    rs = np.random.default_rng(SEED); rs.shuffle(pids)
    folds = np.array_split(pids, k)
    tot, fails = 0.0, 0
    nll = nll_gen("lognormal")
    for f in folds:
        tr, va = data[~data["pid"].isin(f)], data[data["pid"].isin(f)]
        try:
            r = fit_aft(tr, cols)
            tot += nll(r.x, va, Xmat(va, cols))
        except Exception:
            fails += 1
            tot += 1e6
    return tot, fails

base_nll, base_fails = cv_nll(cen, [])
screen = []
kept = []
cur = []
for name, cols in CANDS.items():
    trial = cur + cols
    r, se = fit_se(cen, trial)
    ci_lo = r.x - 1.96 * se; ci_hi = r.x + 1.96 * se
    new_pars = r.x[2:-1][-len(cols):] if cols else np.array([])
    n_new = len(cols)
    lo_new = ci_lo[2 + len(cur):2 + len(cur) + n_new] if n_new else []
    hi_new = ci_hi[2 + len(cur):2 + len(cur) + n_new] if n_new else []
    ci_ok = all((l > 0) or (h < 0) for l, h in zip(lo_new, hi_new)) if n_new else False
    nll_trial, _ = cv_nll(cen, trial)
    nll_cur, _ = cv_nll(cen, cur)
    cv_ok = nll_trial < nll_cur
    est_new = r.x[2 + len(cur):2 + len(cur) + n_new] if n_new else []
    screen.append({"候选": name, "新增项": ";".join(cols), "区间不含0": ci_ok,
                   "估计": ";".join(f"{v:.5f}" for v in est_new),
                   "CI下": ";".join(f"{v:.5f}" for v in lo_new),
                   "CI上": ";".join(f"{v:.5f}" for v in hi_new),
                   "时间比TR": ";".join(f"{np.exp(v):.4f}" for v in est_new),
                   "CV_NLL_现": round(nll_cur, 2),
                   "CV_NLL_加入后": round(nll_trial, 2), "保留": bool(ci_ok and cv_ok)})
    if ci_ok and cv_ok:
        cur = trial
        kept.append(name)
log["协变量筛选"] = screen
log["保留协变量"] = kept
pd.DataFrame(screen).to_csv(os.path.join(OUT, "q3_screen.csv"), index=False, encoding="utf-8-sig")
FINAL_COLS = cur

# ---------------- 4. 最终模型（两分布） ----------------
r_ln, se_ln = fit_se(cen, FINAL_COLS, "lognormal")
r_wb, se_wb = fit_se(cen, FINAL_COLS, "weibull")
aic_ln, aic_wb = 2 * r_ln.fun + 2 * len(r_ln.x), 2 * r_wb.fun + 2 * len(r_wb.x)
dist = "lognormal" if aic_ln <= aic_wb else "weibull"
r_best, se_best = (r_ln, se_ln) if dist == "lognormal" else (r_wb, se_wb)
names = ["alpha0", "alpha1_B0c"] + FINAL_COLS + ["log_sigma"]
rows = []
for mname, r, se, aic in [("lognormal", r_ln, se_ln, aic_ln), ("weibull", r_wb, se_wb, aic_wb)]:
    for k, lab in enumerate(names):
        v = r.x[k]
        rows.append({"分布": mname, "项": lab, "估计": v, "SE": se[k],
                     "CI下": v - 1.96 * se[k], "CI上": v + 1.96 * se[k],
                     "p": float(2 * sps.norm.sf(abs(v / se[k]))), "时间比TR": float(np.exp(v)),
                     "AIC": aic, "采用": mname == dist})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "q3_effects_main.csv"), index=False, encoding="utf-8-sig")
a0, a1 = r_best.x[0], r_best.x[1]
gam = r_best.x[2:-1]; sig = np.exp(r_best.x[-1])
log["AFT采用分布"] = dist
log["最终模型"] = {lab: round(float(v), 5) for lab, v in zip(names, r_best.x)}
log["锚点B_alpha1方向"] = bool(a1 > 0)

TGRID = np.arange(11.0, 25.01, 0.5)
def F_multi(t, data):
    t = np.atleast_1d(t)[:, None]
    mu = a0 + a1 * data["B0c"].values[None, :] + (Xmat(data, FINAL_COLS) @ gam)[None, :]
    z = (np.log(t) - mu) / sig
    return sps.norm.cdf(z) if dist == "lognormal" else 1 - np.exp(-np.exp(z))
F_grid = F_multi(TGRID, cen)

# ---------------- 5. 约束型 DP（同 Q2） ----------------
N = len(cen); MIN_N = 30; P_THRESH = 0.95
order = cen.sort_values("B0")
oidx = order.index.values
B0v = order["B0"].values
P_ord = F_grid[:, oidx]

def earliest_t(pbar, thresh=P_THRESH):
    ok = np.where(pbar >= thresh)[0]
    return (int(ok[0]), True) if len(ok) else (int(np.argmax(pbar)), False)

def optimize_constraint(P, gmax=5, min_n=MIN_N):
    n = P.shape[1]
    Psum = np.hstack([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)])
    @lru_cache(maxsize=None)
    def seg(i, j):
        m = j - i
        pbar = (Psum[:, j] - Psum[:, i]) / m
        k, feas = earliest_t(pbar)
        if not feas:
            return float("inf"), k, feas
        return m * k, k, feas
    results = {}
    for g in range(2, gmax + 1):
        INF = 1e18
        dp = np.full((g + 1, n + 1), INF); pre = np.full((g + 1, n + 1), -1)
        dp[0, 0] = 0.0
        for kk in range(1, g + 1):
            for j in range(min_n * kk, n - min_n * (g - kk) + 1):
                for i in range(min_n * (kk - 1), j - min_n + 1):
                    if dp[kk - 1, i] >= INF:
                        continue
                    c = dp[kk - 1, i] + seg(i, j)[0]
                    if c < dp[kk, j]:
                        dp[kk, j] = c; pre[kk, j] = i
        cuts = []; j = n
        for kk in range(g, 0, -1):
            i = pre[kk, j]; cuts.append((i, j)); j = i
        results[g] = {"cost": dp[g, n], "cuts": cuts[::-1]}
    return results

base = optimize_constraint(P_ord)
mean_weeks = {g: (base[g]["cost"] / N * 0.5 + 11 if base[g]["cost"] < 1e17 else np.inf) for g in base}
log["约束方案各组数平均推荐孕周"] = {g: round(v, 3) for g, v in mean_weeks.items()}
g_chosen = 2
for g in range(3, 6):
    if mean_weeks[g - 1] - mean_weeks[g] >= 0.25:
        g_chosen = g
    else:
        break
log["肘部规则选组数"] = g_chosen

def group_rows(cuts, P, label):
    Psum = np.hstack([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)])
    out = []
    for gi, (i, j) in enumerate(cuts, 1):
        m = j - i
        pbar = (Psum[:, j] - Psum[:, i]) / m
        k, feas = earliest_t(pbar)
        blo = 20.0 if i == 0 else (B0v[i - 1] + B0v[i]) / 2
        bhi = np.inf if j == N else (B0v[j - 1] + B0v[j]) / 2
        out.append({"方案": label, "组": gi,
                    "BMI区间": f"[{blo:.1f}, {bhi:.1f})" if np.isfinite(bhi) else f"[{blo:.1f}, ∞)",
                    "人数": m, "推荐孕周": float(TGRID[k]),
                    "达标比例p(t*)": round(float(pbar[k]), 4), "约束满足": feas})
    return out

cuts_main = base[g_chosen]["cuts"]
main_df = pd.DataFrame(group_rows(cuts_main, P_ord, "Q3多因素约束型主方案(F口径)"))
main_df.to_csv(os.path.join(OUT, "q3_groups_main.csv"), index=False, encoding="utf-8-sig")
log["主方案"] = main_df.to_dict("records")
log["锚点C_低组不晚于高组"] = bool(main_df["推荐孕周"].iloc[0] <= main_df["推荐孕周"].iloc[-1])

# 组间协变量均值（检查 t* 差异来源）
gcov = []
for gi, (i, j) in enumerate(cuts_main, 1):
    sub = order.iloc[i:j]
    gcov.append({"组": gi, "BMI均值": round(float(sub["B0"].mean()), 2),
                 "年龄均值": round(float(sub["age0"].mean()), 2),
                 "身高均值": round(float(sub["height0"].mean()), 2),
                 "IVF占比": round(float((sub["IVF"] == "IVF（试管婴儿）").mean()), 3),
                 "怀孕≥3占比": round(float(sub["preg_3p"].mean()), 3)})
pd.DataFrame(gcov).to_csv(os.path.join(OUT, "q3_group_covariates.csv"), index=False, encoding="utf-8-sig")

# 与 Q2 对照
q2main = pd.read_csv(os.path.join(Q2RES, "q2_groups_main.csv"))
q2f = q2main[q2main["方案"].str.contains("主方案")]
vs = []
for gi in range(1, g_chosen + 1):
    q3r = main_df.iloc[gi - 1]
    q2r = q2f.iloc[gi - 1] if gi - 1 < len(q2f) else None
    vs.append({"组": gi, "Q2_BMI区间": q2r["BMI区间"] if q2r is not None else "",
               "Q2_推荐孕周": q2r["推荐孕周"] if q2r is not None else "",
               "Q3_BMI区间": q3r["BMI区间"], "Q3_推荐孕周": q3r["推荐孕周"],
               "Q3_达标比例": q3r["达标比例p(t*)"]})
vs.append({"组": "平均推荐孕周", "Q2_BMI区间": "", "Q2_推荐孕周": round(float(q2f["推荐孕周"].mean()), 2),
           "Q3_BMI区间": "", "Q3_推荐孕周": round(float(main_df["推荐孕周"].mean()), 2), "Q3_达标比例": ""})
base_cv_ln, _ = cv_nll(cen, FINAL_COLS)
vs.append({"组": "CV删失NLL(小者优)", "Q2_BMI区间": "仅BMI", "Q2_推荐孕周": round(base_nll, 1),
           "Q3_BMI区间": "BMI+" + "+".join(kept), "Q3_推荐孕周": round(base_cv_ln, 1), "Q3_达标比例": ""})
pd.DataFrame(vs).to_csv(os.path.join(OUT, "q3_vs_q2.csv"), index=False, encoding="utf-8-sig")
log["Q3相对Q2"] = {"平均推荐孕周变化": round(float(main_df["推荐孕周"].mean() - q2f["推荐孕周"].mean()), 3),
                     "CV_NLL_仅BMI": round(base_nll, 1), "CV_NLL_多因素": round(base_cv_ln, 1),
                     "CV_NLL增益": round(base_nll - base_cv_ln, 1)}

# ---------------- 6. 误差与分布敏感性 ----------------
def q1coef(term):
    return float(eff1[(eff1["模型"] == "M0") & (eff1["项"] == term)]["估计"].iloc[0])
b0_, b1_, b2_, b4_ = q1coef("Intercept"), q1coef("t_c"), q1coef("B0_c"), q1coef("t_c:B0_c")
gam_ = q1coef("log_reads")
s2_obs = q1coef("sigma_b^2(随机截距方差)") + q1coef("sigma^2(残差方差)")
s2_lat = max(s2_obs - S_HAT ** 2, 1e-8)
lr_med = float(ev1["log_reads"].median())
tc_g = TGRID - 18
m_curve = (b0_ + b1_ * tc_g[:, None] + b2_ * cen["B0c"].values[None, :]
           + b4_ * np.outer(tc_g, cen["B0c"].values) + gam_ * lr_med)
trapz = getattr(np, "trapezoid", None) or np.trapz
def flip_prob(m, sL, sh, c=0.04):
    ys = np.linspace(m - 6 * sL, m + 6 * sL, 2001)
    pdf = sps.norm.pdf(ys, m, sL)
    flip = (ys >= c) * sps.norm.cdf((c - ys) / sh) + (ys < c) * sps.norm.sf((c - ys) / sh)
    return float(trapz(pdf * flip, ys))
sL = np.sqrt(s2_lat)
err_rows = []
for gi, (i, j) in enumerate(cuts_main, 1):
    t_star = float(main_df["推荐孕周"].iloc[gi - 1])
    kt = int(np.argmin(np.abs(TGRID - t_star)))
    flips = [flip_prob(m_curve[kt, w], sL, S_HAT) for w in oidx[i:j]]
    err_rows.append({"组": gi, "指标": "当次误判率(联合正态,Q3时点)", "值": round(float(np.mean(flips)), 4)})
log["当次误判率"] = {r["组"]: r["值"] for r in err_rows}

# 阈值平移 + Weibull 全链条（删失重构沿用 Q2 方法）
ev_q3 = ev1
def build_censoring(threshold):
    rows = []
    for pid, g in ev_q3.sort_values("t_weeks").groupby("pid"):
        t = g["t_weeks"].values
        h = (g["Y"].values >= threshold).astype(int)
        fh = np.argmax(h) if h.any() else None
        if fh is None:
            typ, L, R = "右删失", t[-1], np.inf
        elif fh == 0:
            typ, L, R = "左删失", 0.0, t[0]
        else:
            typ, L, R = "区间删失", t[fh - 1], t[fh]
        rows.append({"pid": pid, "censor_type": typ, "L": L, "R": R})
    out = pd.DataFrame(rows).merge(cen.drop(columns=["censor_type", "L", "R"]), on="pid")
    return out

def chain(dist_name, cen_df, label):
    rb = fit_aft(cen_df, FINAL_COLS, dist_name)
    a0b, a1b = rb.x[0], rb.x[1]
    gamb = rb.x[2:-1]; sigb = np.exp(rb.x[-1])
    mu_b = a0b + a1b * cen_df["B0c"].values[None, :] + (Xmat(cen_df, FINAL_COLS) @ gamb)[None, :]
    z = (np.log(TGRID)[:, None] - mu_b) / sigb
    Fb = sps.norm.cdf(z) if dist_name == "lognormal" else 1 - np.exp(-np.exp(z))
    ob = cen_df.sort_values("B0")
    Fb_ord = Fb[:, ob.index.values]
    res = optimize_constraint(Fb_ord, gmax=2)
    out = []
    for gi, (i, j) in enumerate(res[2]["cuts"], 1):
        pbar = Fb_ord[:, i:j].mean(axis=1)
        k, feas = earliest_t(pbar)
        blo = 20.0 if i == 0 else (B0v[i - 1] + B0v[i]) / 2
        bhi = np.inf if j == N else (B0v[j - 1] + B0v[j]) / 2
        out.append({"情景": label, "组": gi,
                    "BMI区间": f"[{blo:.1f}, {bhi:.1f})" if np.isfinite(bhi) else f"[{blo:.1f}, ∞)",
                    "人数": j - i, "推荐孕周": float(TGRID[k]),
                    "达标比例": round(float(pbar[k]), 4), "整套方案可行": bool(res[2]["cost"] < 1e17)})
    return out

sens = chain("weibull", cen, "Weibull分布")
for sgn, lab in [(-1, "阈值−ŝ"), (1, "阈值+ŝ")]:
    cen_s = build_censoring(0.04 + sgn * S_HAT)
    sens += chain(dist, cen_s, lab)
sens_df = pd.DataFrame(sens)
sens_df.to_csv(os.path.join(OUT, "q3_dist_threshold_sensitivity.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(err_rows).to_csv(os.path.join(OUT, "q3_error_impact.csv"), index=False, encoding="utf-8-sig")
log["分布与阈值敏感性"] = sens_df.to_dict("records")

# ---------------- 7. bootstrap 稳定性 ----------------
B_BOOT = 200
stab, scheme_ok = [], []
for b in range(B_BOOT):
    rng_b = np.random.default_rng(SEED + b)
    bc = cen.iloc[rng_b.choice(len(cen), size=len(cen), replace=True)].reset_index(drop=True)
    try:
        rb = fit_aft(bc, FINAL_COLS, dist)
        a0b, a1b = rb.x[0], rb.x[1]
        gamb = rb.x[2:-1]; sigb = np.exp(rb.x[-1])
        mu_b = a0b + a1b * bc["B0c"].values[None, :] + (Xmat(bc, FINAL_COLS) @ gamb)[None, :]
        z = (np.log(TGRID)[:, None] - mu_b) / sigb
        Fb = sps.norm.cdf(z) if dist == "lognormal" else 1 - np.exp(-np.exp(z))
        ob = bc.sort_values("B0")
        Fb_ord = Fb[:, ob.index.values]
        ob0 = ob["B0"].values
        res = optimize_constraint(Fb_ord, gmax=g_chosen)
        if res[g_chosen]["cost"] >= 1e17:
            raise RuntimeError("不可行")
        cuts_b = res[g_chosen]["cuts"]
        bounds = [round((ob0[j - 1] + ob0[j]) / 2, 1) for (i, j) in cuts_b[:-1]]
        Psum_b = np.hstack([np.zeros((Fb_ord.shape[0], 1)), np.cumsum(Fb_ord, axis=1)])
        feas_all = True
        for gi, (i, j) in enumerate(cuts_b, 1):
            pbar = (Psum_b[:, j] - Psum_b[:, i]) / (j - i)
            k, feas = earliest_t(pbar)
            feas_all = feas_all and feas
            stab.append({"b": b, "组": gi, "t*": float(TGRID[k]),
                         "bounds": ";".join(f"{x:.1f}" for x in bounds), "feas": feas})
        scheme_ok.append(feas_all)
    except Exception:
        scheme_ok.append(False)
    if (b + 1) % 50 == 0:
        print(f"bootstrap {b + 1}/{B_BOOT}")
stab_df = pd.DataFrame(stab)
stab_df.to_csv(os.path.join(OUT, "q3_stability_boot.csv"), index=False, encoding="utf-8-sig")
tstats = stab_df.groupby("组")["t*"].agg(["mean", "std",
    lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)]).round(3)
tstats.columns = ["t*_均值", "t*_SD", "t*_5%", "t*_95%"]
bcnt = Counter()
for s in stab_df["bounds"]:
    for x in s.split(";"):
        bcnt[x] += 1
with open(os.path.join(OUT, "q3_stability.csv"), "w", encoding="utf-8-sig", newline="") as f:
    tstats.to_csv(f)
    f.write("\n# 分界点频次（前10）\n")
    for x, c in bcnt.most_common(10):
        f.write(f"{x},{c}\n")
log["稳定性_t*"] = tstats.to_dict()
log["稳定性_分界点频次前5"] = bcnt.most_common(5)
log["bootstrap整套方案可行率"] = round(float(np.mean(scheme_ok)), 4)
if g_chosen == 2:
    bvals = np.array([float(s) for s in stab_df.loc[stab_df["组"] == 1, "bounds"]])
    log["稳定性_分界点90%区间"] = [round(float(np.quantile(bvals, 0.05)), 1),
                                    round(float(np.quantile(bvals, 0.95)), 1)]

np.save(os.path.join(OUT, "q3_grid_F.npy"), F_grid[:, oidx])
order[["pid", "B0"]].to_csv(os.path.join(OUT, "q3_order.csv"), index=False)
json.dump({"cuts": [[int(i), int(j)] for i, j in cuts_main], "g_chosen": int(g_chosen),
           "kept": kept, "dist": dist}, open(os.path.join(OUT, "q3_cuts.json"), "w"))
with open(os.path.join(OUT, "q3_run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(log, ensure_ascii=False, indent=2, default=str)[:4500])
print("DONE")
