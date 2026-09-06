# -*- coding: utf-8 -*-
"""Q1 主程序：清洗 -> 视图 -> 锚点自检 -> M0/M1 拟合 -> 协变量筛选 -> 按孕妇验证 -> 结果表。
运行: python q1_pipeline.py   (工作目录任意，路径已固定)
输入: 官方附件（只读）  输出: Q1/04_结果/*.csv + q1_run_summary.json
"""
import sys, os, json, re, itertools, warnings
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats as sps

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Q1/
OUT = os.path.join(ROOT, "04_结果")
XLSX = r"D:/10/MCM_China/2025/附件.xlsx"
SEED = 42
os.makedirs(OUT, exist_ok=True)
log = {}

# ---------------- 1. 读取与清洗 ----------------
raw = pd.read_excel(XLSX, sheet_name="男胎检测数据")
raw.columns = [str(c).strip() for c in raw.columns]
log["原始记录数"] = len(raw)

def parse_week(s):
    m = re.match(r"^\s*(\d+)\s*[wW]\s*(?:\+\s*(\d+))?\s*$", str(s))
    if not m:
        return np.nan
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 7.0

df = raw.copy()
df["t_weeks"] = df["检测孕周"].map(parse_week)
log["孕周无法解析条数"] = int(df["t_weeks"].isna().sum())

# BMI 缺失回算并标记
df["bmi_imputed"] = False
miss_bmi = df["孕妇BMI"].isna() & df["身高"].notna() & df["体重"].notna()
df.loc[miss_bmi, "孕妇BMI"] = df.loc[miss_bmi, "体重"] / (df.loc[miss_bmi, "身高"] / 100) ** 2
df.loc[miss_bmi, "bmi_imputed"] = True
log["BMI回算条数"] = int(miss_bmi.sum())

# 完全重复行（除序号外）核验：仅报告，事件均值天然免疫
dup_mask = df.duplicated(subset=[c for c in df.columns if c != "序号"], keep=False)
log["完全重复行条数"] = int(dup_mask.sum())

# 事件键：孕妇代码 + 抽血次数（抽血次数缺失时退回检测日期）
df["抽血次数键"] = df["检测抽血次数"].astype("string").fillna("NA_" + df["检测日期"].astype(str))
df["event_key"] = df["孕妇代码"].astype(str) + "|" + df["抽血次数键"].astype(str)

# 事件内一致性核验：检测日期差 0-6 天视为测序日期；孕周冲突单独标记
def nuniq(s):
    return s.nunique(dropna=True)
chk = df.groupby("event_key").agg(
    n=("序号", "size"), n_week=("t_weeks", nuniq),
    n_age=("年龄", nuniq), n_height=("身高", nuniq), n_weight=("体重", nuniq))
def date_gap(s):
    d = pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return (d.max() - d.min()).days if d.notna().all() else -1
gaps = df.groupby("event_key")["检测日期"].apply(date_gap)
multi = chk[chk["n"] > 1]
week_conflict_keys = multi[multi["n_week"] > 1].index.tolist()
verified = multi[(multi["n_week"] == 1) & (multi["n_age"] == 1) &
                 (multi["n_height"] == 1) & (multi["n_weight"] == 1)]
log["事件总数"] = int(len(chk))
log["多记录事件数"] = int(len(multi))
log["多记录事件涉及行数"] = int(multi["n"].sum())
log["核验为同次采血的事件数"] = int(len(verified) + len(week_conflict_keys))
log["测序日期差分布(天)"] = {int(k): int(v) for k, v in gaps[multi.index].value_counts().sort_index().items()}
log["孕周字段冲突事件"] = week_conflict_keys

# ---------------- 2. 采血级视图 ----------------
def first(s):
    return s.iloc[0]
ev = df.groupby("event_key").agg(
    pid=("孕妇代码", first),
    rows=("序号", lambda s: ";".join(map(str, sorted(s)))),
    t_weeks=("t_weeks", "mean"), t_raw=("检测孕周", first),
    date=("检测日期", first),
    BMI=("孕妇BMI", "mean"), bmi_imputed=("bmi_imputed", "max"),
    Y=("Y染色体浓度", "mean"),
    n_rep=("Y染色体浓度", "size"),
    rep_range=("Y染色体浓度", lambda s: float(s.max() - s.min())),
    age=("年龄", first), height=("身高", first), IVF=("IVF妊娠", first),
    GC=("GC含量", "mean"), map_ratio=("在参考基因组上比对的比例", "mean"),
    dup_ratio=("重复读段的比例", "mean"), uniq_reads=("唯一比对的读段数", "mean"),
    filter_ratio=("被过滤掉读段数的比例", "mean"),
    weight=("体重", "mean"),
).reset_index()
ev["week_conflict"] = ev["event_key"].isin(week_conflict_keys)
ev = ev.sort_values(["pid", "t_weeks"]).reset_index(drop=True)
ev["log_reads"] = np.log10(ev["uniq_reads"])
ev["gc_low"] = ev["GC"] < 0.40
ev["outside_window"] = (ev["t_weeks"] < 10) | (ev["t_weeks"] > 25)
# 基线 BMI 与变化量（按孕周排序的首次事件）
b0 = ev.groupby("pid").first().reset_index()[["pid", "BMI"]].rename(columns={"BMI": "B0"})
ev = ev.merge(b0, on="pid")
ev["dB"] = ev["BMI"] - ev["B0"]
ev["t_c"] = ev["t_weeks"] - 18.0
ev["B0_c"] = ev["B0"] - ev["B0"].mean()
ev["y_hit"] = (ev["Y"] >= 0.04).astype(int)
log["采血级记录数"] = int(len(ev))
log["孕妇数"] = int(ev["pid"].nunique())
log["孕周范围"] = [round(float(ev["t_weeks"].min()), 2), round(float(ev["t_weeks"].max()), 2)]
log["BMI范围"] = [round(float(ev["BMI"].min()), 2), round(float(ev["BMI"].max()), 2)]
log["首次事件即达标人数"] = int(ev.groupby("pid").first()["y_hit"].sum())
log["观测窗口外事件数"] = int(ev["outside_window"].sum())
log["GC低于40%事件数"] = int(ev["gc_low"].sum())
ev.to_csv(os.path.join(OUT, "q1_blood_events.csv"), index=False, encoding="utf-8-sig")

# ---------------- 3. 技术重复视图与误差估计 ----------------
reps = []
for key, g in df.groupby("event_key"):
    ys = g["Y染色体浓度"].dropna().values
    if len(ys) >= 2:
        diffs = [round(float(a - b), 6) for a, b in itertools.combinations(ys, 2)]
        reps.append({"pid": g["孕妇代码"].iloc[0], "event_key": key, "n_rep": len(ys),
                     "Y_values": ";".join(f"{v:.6f}" for v in ys),
                     "pairwise_diffs": ";".join(map(str, diffs)),
                     "range": float(ys.max() - ys.min())})
rep_df = pd.DataFrame(reps)
rep_df.to_csv(os.path.join(OUT, "q1_technical_replicates.csv"), index=False, encoding="utf-8-sig")
all_d = np.array([d for r in reps for d in r["pairwise_diffs"].split(";")], dtype=float)
s_hat = float(all_d.std(ddof=1) / np.sqrt(2))
log["技术重复事件数"] = int(len(reps))
log["重复差中位绝对值"] = round(float(np.median(np.abs(all_d))), 6)
log["单次测量误差估计s_hat"] = round(s_hat, 6)
log["s_hat相对4%阈值"] = round(s_hat / 0.04, 4)

# ---------------- 4. 锚点自检 ----------------
# 锚点1: sigma_b^2 = 0 时 MixedLM 应退化为 OLS
rng = np.random.default_rng(0)
n = 400
xs = rng.normal(size=n)
pid_s = np.repeat(np.arange(100), 4)
y_s = 1.5 + 2.0 * xs + rng.normal(scale=0.5, size=n)   # 无随机截距
d_s = pd.DataFrame({"y": y_s, "x": xs, "pid": pid_s})
b_ols = smf.ols("y ~ x", data=d_s).fit().params.values
b_mix = smf.mixedlm("y ~ x", data=d_s, groups=d_s["pid"]).fit(reml=False).params.values[:2]
anchor1 = bool(np.max(np.abs(b_ols - b_mix)) < 1e-3)
log["锚点1_MixedLM退化为OLS"] = anchor1
assert anchor1, f"锚点1失败: OLS={b_ols}, MixedLM={b_mix}"

# ---------------- 5. 样条基与建模工具 ----------------
def spline_cols(data, design_info=None):
    """自然三次样条基 df=3；design_info 用于训练/验证一致展开。"""
    if design_info is None:
        dm = patsy.dmatrix("0 + cr(t_c, df=3)", data, return_type="dataframe")
        design_info = dm.design_info
    else:
        dm = patsy.build_design_matrices([design_info], data)[0]
    out = pd.DataFrame(np.asarray(dm), index=data.index,
                       columns=[f"sp{k}" for k in range(dm.shape[1])])
    return out, design_info

def fit(formula, data, reml):
    return smf.mixedlm(formula, data=data, groups=data["pid"]).fit(reml=reml, method="lbfgs")

CORE = "Y ~ t_c + B0_c + dB + t_c:B0_c"
def m1_rhs(extra=""):
    return "Y ~ sp0 + sp1 + sp2 + B0_c + dB + t_c:B0_c" + extra + " - 1"  # 样条基已张成常数项，去掉截距避免共线

# ---------------- 6. 全数据拟合 M0/M1 与 LRT ----------------
ev_m = ev.dropna(subset=["Y", "t_c", "B0_c", "dB"]).copy()
sp_all, sp_info = spline_cols(ev_m)
ev_m = pd.concat([ev_m, sp_all], axis=1)

m0_ml, m0_reml = fit(CORE, ev_m, False), fit(CORE, ev_m, True)
m1_ml, m1_reml = fit(m1_rhs(), ev_m, False), fit(m1_rhs(), ev_m, True)
m_noweek_ml = fit("Y ~ B0_c + dB", ev_m, False)

lrt_nonlin = 2 * (m1_ml.llf - m0_ml.llf)          # df=1：样条较线性的非线性分量
lrt_week = 2 * (m1_ml.llf - m_noweek_ml.llf)      # df=3：孕周整体效应
log["LRT_孕周非线性分量"] = {"chi2": round(float(lrt_nonlin), 3), "df": 1,
                             "p": float(sps.chi2.sf(lrt_nonlin, 1))}
log["LRT_孕周整体效应"] = {"chi2": round(float(lrt_week), 3), "df": 3,
                           "p": float(sps.chi2.sf(lrt_week, 3))}

# ---------------- 7. 协变量筛选（先 CI 后 CV） ----------------
CANDS = {"age": "+ age", "GC": "+ GC", "map_ratio": "+ map_ratio",
         "filter_ratio": "+ filter_ratio", "log_reads": "+ log_reads",
         "IVF": "+ C(IVF)"}
def woman_cv(formula_builder, data, spline=False, k=5):
    """按孕妇 5 折：孕妇等权 MAE/RMSE + 记录级 MAE/RMSE。"""
    pids = np.array(sorted(data["pid"].unique()))
    rs = np.random.default_rng(SEED)
    rs.shuffle(pids)
    folds = np.array_split(pids, k)
    wmae, wrmse, rmae, rrmse = [], [], [], []
    for f in folds:
        tr = data[~data["pid"].isin(f)].copy(); va = data[data["pid"].isin(f)].copy()
        tr = tr.loc[:, ~tr.columns.str.match('sp[0-9]+')]; va = va.loc[:, ~va.columns.str.match('sp[0-9]+')]
        if spline:
            sp_tr, info = spline_cols(tr)
            sp_va, _ = spline_cols(va, info)
            tr = pd.concat([tr, sp_tr], axis=1)
            va = pd.concat([va, sp_va], axis=1)
        try:
            r = fit(formula_builder(), tr, reml=False)
        except Exception:
            r = fit(formula_builder(), tr, reml=False)
        pred = r.predict(exog=va)
        err = va["Y"].values - np.asarray(pred)
        g = pd.DataFrame({"pid": va["pid"].values, "e": err})
        per_w = g.groupby("pid")["e"].agg(["mean", "size"])
        per_w_mae = g.assign(ae=g["e"].abs()).groupby("pid")["ae"].mean()
        per_w_rmse = g.assign(se=g["e"] ** 2).groupby("pid")["se"].mean().pow(0.5)
        wmae.append(float(per_w_mae.mean())); wrmse.append(float(np.sqrt((per_w_rmse ** 2).mean())))
        rmae.append(float(np.abs(err).mean())); rrmse.append(float(np.sqrt((err ** 2).mean())))
    return {"wMAE": np.mean(wmae), "wRMSE": np.mean(wrmse),
            "rMAE": np.mean(rmae), "rRMSE": np.mean(rrmse)}

base_formula = CORE
kept = []
screen_rows = []
cv_cache = {(): woman_cv(lambda: CORE, ev_m, spline=False)}
for name, term in CANDS.items():
    try:
        r = fit(CORE + term, ev_m, False)
        ci = r.conf_int()
        terms = [t for t in r.params.index if t not in m0_ml.params.index]
        ci_ok = all((ci.loc[t, 0] > 0) or (ci.loc[t, 1] < 0) for t in terms)
    except Exception as e:
        ci_ok, terms = False, [f"拟合失败:{e}"]
    screen_rows.append({"协变量": name, "新增项": ";".join(terms), "CI不含0": ci_ok})
    if ci_ok:
        key = tuple(sorted(kept + [name]))
        f_new = CORE + "".join(CANDS[k] for k in sorted(kept + [name]))
        cv0 = cv_cache[tuple(sorted(kept))] if tuple(sorted(kept)) in cv_cache else \
            woman_cv((lambda f: (lambda: f))(CORE + "".join(CANDS[k] for k in sorted(kept))), ev_m)
        cv1 = woman_cv((lambda f: (lambda: f))(f_new), ev_m)
        screen_rows[-1]["wRMSE_现模型"] = round(cv0["wRMSE"], 6)
        screen_rows[-1]["wRMSE_加入后"] = round(cv1["wRMSE"], 6)
        if cv1["wRMSE"] < cv0["wRMSE"]:
            kept.append(name)
            cv_cache[key] = cv1
for k in [()]:
    cv_cache.setdefault((), woman_cv(lambda: CORE, ev_m))
log["协变量保留"] = kept
EXTRA = "".join(CANDS[k] for k in sorted(kept))
log["最终附加项"] = EXTRA if EXTRA else "(无)"
pd.DataFrame(screen_rows).to_csv(os.path.join(OUT, "q1_covariate_screen.csv"),
                                 index=False, encoding="utf-8-sig")

# 最终模型（含保留协变量），ML 比较 + REML 报告
f0 = CORE + EXTRA
f1 = m1_rhs(EXTRA)
m0_ml, m0_reml = fit(f0, ev_m, False), fit(f0, ev_m, True)
m1_ml, m1_reml = fit(f1, ev_m, False), fit(f1, ev_m, True)
m_noweek_ml = fit("Y ~ B0_c + dB" + EXTRA, ev_m, False)
lrt_nonlin = 2 * (m1_ml.llf - m0_ml.llf)
lrt_week = 2 * (m1_ml.llf - m_noweek_ml.llf)
log["最终_LRT_孕周非线性分量"] = {"chi2": round(float(lrt_nonlin), 3), "df": 1,
                                 "p": float(sps.chi2.sf(lrt_nonlin, 1))}
log["最终_LRT_孕周整体效应"] = {"chi2": round(float(lrt_week), 3), "df": 3,
                               "p": float(sps.chi2.sf(lrt_week, 3))}

# ---------------- 8. 按孕妇验证 M0 vs M1 ----------------
cv0 = woman_cv((lambda f: (lambda: f))(f0), ev_m, spline=False)
cv1 = woman_cv((lambda f: (lambda: f))(f1), ev_m, spline=True)
impr = (cv0["wRMSE"] - cv1["wRMSE"]) / cv0["wRMSE"]
log["验证_M0"] = {k: round(v, 6) for k, v in cv0.items()}
log["验证_M1"] = {k: round(v, 6) for k, v in cv1.items()}
log["M1相对M0_wRMSE改善"] = round(float(impr), 4)
keep_m1 = bool(impr > 0.02 and sps.chi2.sf(lrt_nonlin, 1) < 0.05)
chosen, chosen_res = ("M1", m1_reml) if keep_m1 else ("M0", m0_reml)
log["保留规则结论"] = f"采用 {chosen}（改善={impr:.2%}, 非线性p={sps.chi2.sf(lrt_nonlin,1):.4g}）"

# logit 尺度对照（检验高斯尺度稳健性；Duan smearing 回变换）
def woman_cv_logit(formula, data, k=5):
    pids = np.array(sorted(data["pid"].unique()))
    rs = np.random.default_rng(SEED)
    rs.shuffle(pids)
    folds = np.array_split(pids, k)
    wmae, wrmse = [], []
    logistic = lambda x: 1.0 / (1.0 + np.exp(-x))
    for f in folds:
        tr, va = data[~data["pid"].isin(f)].copy(), data[data["pid"].isin(f)].copy()
        r = fit(formula, tr, reml=False)
        lp = np.asarray(r.predict(exog=va))
        e_tr = tr["logitY"].values - np.asarray(r.fittedvalues)
        pred = logistic(lp[:, None] + e_tr[None, :]).mean(axis=1)   # Duan smearing
        err = va["Y"].values - pred
        g = pd.DataFrame({"pid": va["pid"].values, "e": err})
        wmae.append(float(g.assign(ae=g["e"].abs()).groupby("pid")["ae"].mean().mean()))
        wrmse.append(float(np.sqrt(g.assign(se=g["e"] ** 2).groupby("pid")["se"].mean().mean())))
    return {"wMAE": np.mean(wmae), "wRMSE": np.mean(wrmse)}

ev_m["logitY"] = np.log(ev_m["Y"] / (1 - ev_m["Y"]))
f0l = f0.replace("Y ~", "logitY ~")
cv_logit = woman_cv_logit(f0l, ev_m)
log["验证_M0_logit尺度对照"] = {k: round(v, 6) for k, v in cv_logit.items()}
log["logit对照结论"] = ("logit 尺度更优，需复核尺度选择" if cv_logit["wRMSE"] < cv0["wRMSE"] * 0.98
                       else "logit 尺度无稳定收益，维持原始尺度高斯模型")

# ---------------- 9. 效应主表 ----------------
def r2_lmm(res, data):
    b_map = data['pid'].map(lambda p: float(np.asarray(res.random_effects[p])[0]))
    fe_var = float(np.var(np.asarray(res.fittedvalues) - b_map.values))
    vb = float(res.cov_re.iloc[0, 0]); ve = float(res.scale)
    return fe_var / (fe_var + vb + ve), (fe_var + vb) / (fe_var + vb + ve)

rows = []
for name, res in [("M0", m0_reml), ("M1", m1_reml)]:
    ci = res.conf_int()
    for t in res.params.index:
        if t in ("Group Var",):
            continue
        rows.append({"模型": name, "项": t, "估计": res.params[t],
                     "标准误": res.bse[t], "CI下": ci.loc[t, 0], "CI上": ci.loc[t, 1],
                     "z": res.tvalues[t], "p": res.pvalues[t]})
    vb, ve = float(res.cov_re.iloc[0, 0]), float(res.scale)
    r2m, r2c = r2_lmm(res, ev_m)
    rows += [{"模型": name, "项": "sigma_b^2(随机截距方差)", "估计": vb},
             {"模型": name, "项": "sigma^2(残差方差)", "估计": ve},
             {"模型": name, "项": "ICC", "估计": vb / (vb + ve)},
             {"模型": name, "项": "边际R2", "估计": r2m},
             {"模型": name, "项": "条件R2", "估计": r2c}]
rows += [{"模型": "LRT", "项": "孕周非线性分量(M1 vs M0, df=1)",
          "估计": float(lrt_nonlin), "p": float(sps.chi2.sf(lrt_nonlin, 1))},
         {"模型": "LRT", "项": "孕周整体效应(M1 vs 无孕周, df=3)",
          "估计": float(lrt_week), "p": float(sps.chi2.sf(lrt_week, 3))}]
eff = pd.DataFrame(rows)
eff.to_csv(os.path.join(OUT, "q1_effects_main.csv"), index=False, encoding="utf-8-sig")

val = pd.DataFrame([
    {"模型": "M0(LMM线性)", **{k: round(v, 6) for k, v in cv0.items()}},
    {"模型": "M1(样条df=3)", **{k: round(v, 6) for k, v in cv1.items()}},
    {"模型": "M0-logit尺度对照(smearing回变换)", **{k: round(v, 6) for k, v in cv_logit.items()}},
    {"模型": "结论", "wMAE": log["保留规则结论"] + "；" + log["logit对照结论"]}])
val.to_csv(os.path.join(OUT, "q1_validation.csv"), index=False, encoding="utf-8-sig")

# ---------------- 10. 效应曲线与锚点2/3 ----------------
grid = pd.DataFrame({"t_c": np.linspace(ev_m["t_c"].min(), ev_m["t_c"].max(), 120)})
grid["B0_c"] = 0.0; grid["dB"] = 0.0; grid["pid"] = ev_m["pid"].iloc[0]
for c in kept:
    grid[c] = ev_m[c].median() if ev_m[c].dtype != object else ev_m[c].mode()[0]
sp_g, _ = spline_cols(grid, sp_info)
grid = pd.concat([grid, sp_g], axis=1)
curve = pd.DataFrame({"t_weeks": grid["t_c"] + 18})
for name, res in [("M0", m0_reml), ("M1", m1_reml)]:
    Xg = np.asarray(patsy.dmatrix(res.model.formula.split("~", 1)[1], grid))
    fe = res.fe_params.values
    cov = res.cov_params().loc[res.fe_params.index, res.fe_params.index].values
    pred = Xg @ fe
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov, Xg))
    curve[f"{name}_pred"] = pred
    curve[f"{name}_lo"] = pred - 1.96 * se
    curve[f"{name}_hi"] = pred + 1.96 * se
curve.to_csv(os.path.join(OUT, "q1_effect_curve.csv"), index=False, encoding="utf-8-sig")

b1 = m0_reml.params.get("t_c", np.nan); b2 = m0_reml.params.get("B0_c", np.nan)
log["锚点2_方向检查"] = {"beta1_t_c": round(float(b1), 6), "beta2_B0_c": round(float(b2), 6),
                         "符合先验(beta1>0,beta2<0)": bool(b1 > 0 and b2 < 0)}
in_grid = curve[(curve["t_weeks"] >= 11) & (curve["t_weeks"] <= 25)][f"{chosen}_pred"]
log["锚点3_预测范围"] = {"周内11-25最小": round(float(in_grid.min()), 5),
                         "周内11-25最大": round(float(in_grid.max()), 5),
                         "在(0,0.15)内": bool(in_grid.min() > 0 and in_grid.max() < 0.15)}

# 残差诊断数据（供绘图）
dg = ev_m[["pid", "t_weeks", "B0", "dB", "Y"]].copy()
dg["fitted"] = chosen_res.fittedvalues.values
dg["resid"] = dg["Y"] - dg["fitted"]
dg["model"] = chosen
dg.to_csv(os.path.join(OUT, "q1_diagnostics.csv"), index=False, encoding="utf-8-sig")

# logit 尺度对照（残差偏态触发时人工查看诊断图再决定；此处仅记录偏度）
log["残差偏度"] = round(float(sps.skew(dg["resid"])), 3)

with open(os.path.join(OUT, "q1_run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(log, ensure_ascii=False, indent=2, default=str))
print("DONE")
