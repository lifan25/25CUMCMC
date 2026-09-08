# -*- coding: utf-8 -*-
"""Q4 主程序 v2（按队长评审修正）：阈值严格折内选择、bootstrap 保留 multiplicity、
标签口径三分量、可落地模型规格、消融对照、事件级敏感性。
运行: python q4_pipeline.py [--xlsx 附件路径]   输出: Q4/04_结果/
"""
import sys, os, json, argparse, re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
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

# ---------------- 1. 数据与标签口径 ----------------
def parse_week(s):
    m = re.match(r"^\s*(\d+)\s*[wW]\s*(?:\+\s*(\d+))?\s*$", str(s))
    return np.nan if not m else int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 7.0

df = pd.read_excel(XLSX, sheet_name="女胎检测数据")
df.columns = [str(c).strip() for c in df.columns]
df["t_weeks"] = df["检测孕周"].map(parse_week)
miss_bmi = df["孕妇BMI"].isna()
df.loc[miss_bmi, "孕妇BMI"] = df.loc[miss_bmi, "体重"] / (df.loc[miss_bmi, "身高"] / 100) ** 2
ab = df["染色体的非整倍体"].fillna("")
for k in ["T13", "T18", "T21"]:
    df[f"y_{k}"] = ab.str.contains(k).astype(int)
LABELS = ["T13", "T18", "T21"]
ZCOL = {"T13": "13号染色体的Z值", "T18": "18号染色体的Z值", "T21": "21号染色体的Z值"}
log["记录数/孕妇数"] = [len(df), int(df["孕妇代码"].nunique())]
log["阳性记录数"] = {k: int(df[f"y_{k}"].sum()) for k in LABELS}
wlab = df.groupby("孕妇代码")[[f"y_{k}" for k in LABELS]].max()
log["孕妇级阳性数"] = {k: int(wlab[f"y_{k}"].sum()) for k in LABELS}
log["BMI回算条数"] = int(miss_bmi.sum())

# 标签口径三分量（队长修正）：非空标签间变化 / 空白↔阳性变化 / 事件内不一致
ch_nonblank = df.groupby("孕妇代码")["染色体的非整倍体"].apply(lambda s: len(set(s.dropna())) > 1).sum()
gp = df.groupby("孕妇代码")["染色体的非整倍体"]
ch_blank_pos = gp.apply(lambda s: (s.isna().any()) and (s.notna().any())).sum()
df["event_key"] = df["孕妇代码"].astype(str) + "|" + df["检测抽血次数"].astype(str)
ev_incons = df.groupby("event_key")["染色体的非整倍体"].apply(
    lambda s: len(set(s.fillna("(空白)"))) > 1).sum()
log["标签口径"] = {"非空标签间变化孕妇数": int(ch_nonblank),
                    "空白与阳性并存孕妇数": int(ch_blank_pos),
                    "事件内标签不一致事件数": int(ev_incons),
                    "说明": "标签是逐记录检测结论；记录级复现与孕妇真实状态须区分"}
log["采血事件数"] = int(df["event_key"].nunique())

FEATS = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "X染色体浓度",
         "GC含量", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
         "log_reads", "log_uniq", "在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例",
         "孕妇BMI", "年龄", "身高", "t_weeks"]
df["log_reads"] = np.log10(df["原始读段数"])
df["log_uniq"] = np.log10(df["唯一比对的读段数"])
X = df[FEATS].copy()
groups = df["孕妇代码"].values
df[FEATS + ["y_T13", "y_T18", "y_T21", "孕妇代码", "event_key"]].to_csv(
    os.path.join(OUT, "q4_features_labels.csv"), index=False, encoding="utf-8-sig")

# ---------------- 2. 模型族 ----------------
def m1_fn(Xtr, ytr, gtr):
    inner = GroupKFold(3)
    best_c, best_s = 0.1, -1
    for c in [0.03, 0.1, 0.3]:
        s = []
        for itr, iva in inner.split(Xtr, ytr, gtr):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
            m.fit(Xtr.iloc[itr], ytr[itr])
            pv = m.predict_proba(Xtr.iloc[iva])[:, 1]
            if ytr[iva].sum() > 0:
                s.append(average_precision_score(ytr[iva], pv))
        if s and np.mean(s) > best_s:
            best_s, best_c = np.mean(s), c
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000))
    m.fit(Xtr, ytr)
    return m, best_c

def m2_fn(Xtr, ytr, gtr):
    m = make_pipeline(StandardScaler(),
                      DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                             class_weight="balanced", random_state=SEED))
    m.fit(Xtr, ytr)
    return m, None

gkf = GroupKFold(5)

# 锚点 A：打乱标签应无信号
rng = np.random.default_rng(0)
auc_shuf = []
for k in LABELS:
    y_sh = rng.permutation(df[f"y_{k}"].values)
    for tr, va in gkf.split(X, y_sh, groups):
        m, _ = m1_fn(X.iloc[tr], y_sh[tr], groups[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        if 0 < y_sh[va].sum() < len(va):
            auc_shuf.append(roc_auc_score(y_sh[va], p))
anchorA = abs(float(np.mean(auc_shuf)) - 0.5) < 0.1
log["锚点A_打乱标签CV_AUC(应≈0.5)"] = {"均值": round(float(np.mean(auc_shuf)), 3), "通过": anchorA}
assert anchorA

# ---------------- 3. 规则基线（|Z|>3 与全范围数据驱动 τ） ----------------
base_rows = []
for k in LABELS:
    y = df[f"y_{k}"].values
    z = df[ZCOL[k]].abs().values
    pred = z > 3
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int(((~pred) & (y == 1)).sum())
    base_rows.append({"染色体": k, "规则": "|Z|>3（教科书）", "TP": tp, "FP": fp, "FN": fn,
                      "召回": round(tp / max(tp + fn, 1), 4), "精确率": round(tp / max(tp + fp, 1), 4)})
    oof = np.zeros(len(df))
    for tr, va in gkf.split(X, y, groups):
        cand = np.unique(z[tr])          # 全范围候选（队长修正）
        f1s = []
        for t in cand:
            pr = z[tr] > t
            tp_ = (pr & (y[tr] == 1)).sum(); fp_ = (pr & (y[tr] == 0)).sum(); fn_ = ((~pr) & (y[tr] == 1)).sum()
            f1s.append(2 * tp_ / max(2 * tp_ + fp_ + fn_, 1))
        oof[va] = (z[va] > cand[int(np.argmax(f1s))]).astype(float)
    tp, fp, fn = int((oof * y).sum()), int((oof * (1 - y)).sum()), int(((1 - oof) * y).sum())
    base_rows.append({"染色体": k, "规则": "|Z|>τ（训练折全范围 F1 选 τ）", "TP": tp, "FP": fp, "FN": fn,
                      "召回": round(tp / max(tp + fn, 1), 4), "精确率": round(tp / max(tp + fp, 1), 4)})
base_df = pd.DataFrame(base_rows)
base_df.to_csv(os.path.join(OUT, "q4_baselines.csv"), index=False, encoding="utf-8-sig")
log["锚点B_Z规则基线复现"] = base_rows[0]["召回"] < 0.1 and base_rows[2]["召回"] < 0.1 and base_rows[4]["召回"] < 0.1
assert log["锚点B_Z规则基线复现"]

# ---------------- 4. 模型对照（OOF 排序指标） + 消融 ----------------
def oof_predict(model_fn, y, Xuse):
    oof = np.zeros(len(Xuse))
    for tr, va in gkf.split(Xuse, y, groups):
        m, _ = model_fn(Xuse.iloc[tr], y[tr], groups[tr])
        oof[va] = m.predict_proba(Xuse.iloc[va])[:, 1]
    return oof

oof_m1, oof_m2, cmp_rows = {}, {}, []
for k in LABELS:
    y = df[f"y_{k}"].values
    oof_m1[k] = oof_predict(m1_fn, y, X)
    oof_m2[k] = oof_predict(m2_fn, y, X)
    for name, oof in [("M1_逻辑回归", oof_m1[k]), ("M2_浅层树", oof_m2[k])]:
        cmp_rows.append({"染色体": k, "模型": name,
                         "ROC_AUC": round(roc_auc_score(y, oof), 4),
                         "PR_AUC": round(average_precision_score(y, oof), 4),
                         "随机水平": round(float(y.mean()), 4)})
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(OUT, "q4_model_compare.csv"), index=False, encoding="utf-8-sig")
use_tree = {k: bool(cmp_df[(cmp_df["染色体"] == k) & (cmp_df["模型"] == "M2_浅层树")]["PR_AUC"].iloc[0]
                    - cmp_df[(cmp_df["染色体"] == k) & (cmp_df["模型"] == "M1_逻辑回归")]["PR_AUC"].iloc[0] > 0.02)
            for k in LABELS}
log["模型比较"] = cmp_rows
log["M2保留"] = use_tree

# 消融：同一分组折比较特征子集（队长建议）
def feat_sets(k):
    z4 = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"]
    qual = ["GC含量", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
            "log_reads", "log_uniq", "在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例"]
    return {"A_对应Z单变量": [ZCOL[k]], "B_全部Z值": z4,
            "C_Z+GC与读段质量": z4 + qual, "D_完整18因素": FEATS}
abl_rows = []
for k in LABELS:
    y = df[f"y_{k}"].values
    for sname, fs in feat_sets(k).items():
        oof = oof_predict(m1_fn, y, X[fs])
        abl_rows.append({"染色体": k, "特征集": sname, "特征数": len(fs),
                         "ROC_AUC": round(roc_auc_score(y, oof), 4),
                         "PR_AUC": round(average_precision_score(y, oof), 4)})
abl_df = pd.DataFrame(abl_rows)
abl_df.to_csv(os.path.join(OUT, "q4_ablation.csv"), index=False, encoding="utf-8-sig")
log["消融对照"] = abl_rows

# ---------------- 5. 阈值：严格外层折内选择（队长 P1-1/P1-2） ----------------
SCEN = {"偏精确(召回≥0.70)": 0.70, "基准(召回≥0.80)": 0.80, "偏召回(召回≥0.90)": 0.90}

def pick_threshold(p_inner, y_inner, target):
    """满足 内层召回>=target 的最高阈值（精确率最优）；target 总可由低阈值达到。"""
    cand = np.unique(p_inner)
    feas = [t for t in cand if (p_inner >= t)[y_inner == 1].mean() >= target]
    return float(max(feas)) if feas else 0.0

def metrics_at(y, pred_bin):
    pred_bin = pred_bin.astype(bool); yb = y.astype(bool)
    tp = int((pred_bin & yb).sum()); fp = int((pred_bin & ~yb).sum())
    fn = int((~pred_bin & yb).sum()); tn = int((~pred_bin & ~yb).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "召回": round(tp / max(tp + fn, 1), 4), "精确率": round(tp / max(tp + fp, 1), 4),
            "特异度": round(tn / max(tn + fp, 1), 4), "F1": round(2 * tp / max(2 * tp + fp + fn, 1), 4)}

perf_rows = []
oof_final = {k: np.zeros(len(X)) for k in LABELS}       # 外层模型概率（供 bootstrap/事件级）
bin_final = {k: {s: np.zeros(len(X), dtype=int) for s in SCEN} for k in LABELS}
thresh_median = {k: {} for k in LABELS}
for k in LABELS:
    y = df[f"y_{k}"].values
    model_fn = m2_fn if use_tree[k] else m1_fn
    th_fold = {s: [] for s in SCEN}
    for tr, va in gkf.split(X, y, groups):
        m, _ = model_fn(X.iloc[tr], y[tr], groups[tr])
        p_va = m.predict_proba(X.iloc[va])[:, 1]
        oof_final[k][va] = p_va
        # 内层 OOF 仅用外层训练折（阈值选择不得见验证折）
        p_inner = np.zeros(len(tr))
        inner = GroupKFold(3)
        for itr, iva in inner.split(X.iloc[tr], y[tr], groups[tr]):
            mi, _ = model_fn(X.iloc[tr].iloc[itr], y[tr][itr], groups[tr][itr])
            p_inner[iva] = mi.predict_proba(X.iloc[tr].iloc[iva])[:, 1]
        for s, target in SCEN.items():
            th = pick_threshold(p_inner, y[tr], target)
            th_fold[s].append(th)
            bin_final[k][s][va] = (p_va >= th).astype(int)
    for s in SCEN:
        thresh_median[k][s] = round(float(np.median(th_fold[s])), 4)
        m_ = metrics_at(y, bin_final[k][s])
        perf_rows.append({"染色体": k, "情景": s, "阈值(折中位)": thresh_median[k][s],
                          "层级": "记录级", **m_})
        # 孕妇级：记录二元判定按事件折阈值，孕妇取任一记录判阳
        wy = pd.Series(y).groupby(groups).max()
        wp_bin = pd.Series(bin_final[k][s]).groupby(groups).max()
        m_w = metrics_at(wy.values, wp_bin.values)
        perf_rows.append({"染色体": k, "情景": s, "阈值(折中位)": thresh_median[k][s],
                          "层级": "孕妇级", **m_w})
perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(os.path.join(OUT, "q4_performance.csv"), index=False, encoding="utf-8-sig")
log["阈值(折中位)"] = thresh_median
log["性能_基准_记录级"] = [r for r in perf_rows if r["情景"].startswith("基准") and r["层级"] == "记录级"]
log["性能_基准_孕妇级"] = [r for r in perf_rows if r["情景"].startswith("基准") and r["层级"] == "孕妇级"]
log["性能_偏召回_记录级"] = [r for r in perf_rows if r["情景"].startswith("偏召回") and r["层级"] == "记录级"]

# ---------------- 6. bootstrap（保留 multiplicity；固定 OOF 的评价不确定性） ----------------
B = 500
pid_list = df["孕妇代码"].unique()
idx_by_pid = {p: np.where(groups == p)[0] for p in pid_list}
boot_rows = []
for k in LABELS:
    y = df[f"y_{k}"].values
    oof = oof_final[k]
    th = thresh_median[k]["基准(召回≥0.80)"]
    recs, pras = [], []
    for b in range(B):
        samp = np.random.default_rng(SEED + b).choice(pid_list, size=len(pid_list), replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in samp])   # 保留重复抽到的 multiplicity
        if y[idx].sum() == 0:
            continue
        pred = (oof[idx] >= th).astype(int)
        tp = (pred & (y[idx] == 1)).sum(); fn = ((1 - pred) & (y[idx] == 1)).sum()
        recs.append(tp / max(tp + fn, 1))
        pras.append(average_precision_score(y[idx], oof[idx]))
    boot_rows.append({"染色体": k, "召回_中位": round(float(np.median(recs)), 4),
                      "召回_5%": round(float(np.quantile(recs, 0.05)), 4),
                      "召回_95%": round(float(np.quantile(recs, 0.95)), 4),
                      "PRAUC_中位": round(float(np.median(pras)), 4),
                      "PRAUC_5%": round(float(np.quantile(pras, 0.05)), 4),
                      "PRAUC_95%": round(float(np.quantile(pras, 0.95)), 4)})
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(os.path.join(OUT, "q4_bootstrap.csv"), index=False, encoding="utf-8-sig")
log["bootstrap区间(基准,记录级)"] = boot_rows
log["bootstrap口径"] = "对固定 OOF 预测重抽孕妇（保留 multiplicity），估计固定模型评价样本的不确定性，非重拟合稳定性"

# ---------------- 7. 事件级敏感性（技术重复加权） ----------------
ev_rows = []
ev_key = df["event_key"].values
for k in LABELS:
    y = df[f"y_{k}"].values
    oof = oof_final[k]
    ev_y = pd.Series(y).groupby(ev_key).max()
    ev_p = pd.Series(oof).groupby(ev_key).max()
    th = thresh_median[k]["基准(召回≥0.80)"]
    m_ = metrics_at(ev_y.values, (ev_p.values >= th).astype(int))
    ev_rows.append({"染色体": k, "事件数": len(ev_y),
                    "ROC_AUC": round(roc_auc_score(ev_y, ev_p), 4),
                    "PR_AUC": round(average_precision_score(ev_y, ev_p), 4),
                    "基准召回": m_["召回"], "基准精确率": m_["精确率"]})
ev_df = pd.DataFrame(ev_rows)
ev_df.to_csv(os.path.join(OUT, "q4_event_sensitivity.csv"), index=False, encoding="utf-8-sig")
log["事件级敏感性"] = ev_rows

# ---------------- 8. 可落地判定规则（完整规格） ----------------
spec = {}
coef_rows = []
for k in LABELS:
    y = df[f"y_{k}"].values
    m, best_c = m1_fn(X, y, groups)          # 全数据按同一内层规则选 C
    sc = m.named_steps["standardscaler"]
    lr = m.named_steps["logisticregression"]
    spec[k] = {"C": best_c, "intercept": float(lr.intercept_[0]),
               "features": [{"name": f_, "mean": float(mu), "std": float(sd),
                             "coef": float(c_)}
                            for f_, mu, sd, c_ in zip(FEATS, sc.mean_, sc.scale_, lr.coef_[0])],
               "thresholds": thresh_median[k],
               "formula": "P = 1/(1+exp(-(intercept + sum(coef_i * (x_i - mean_i)/std_i))))",
               "label_rule": "P >= thresholds[情景] 判该染色体异常；组合标签由三判定合成；任一判阳建议复检"}
    for f_, c_ in sorted(zip(FEATS, lr.coef_[0]), key=lambda t: -abs(t[1])):
        coef_rows.append({"染色体": k, "特征": f_, "标准化系数": round(float(c_), 4)})
    if k == "T18":
        zc = dict(zip(FEATS, lr.coef_[0]))[ZCOL["T18"]]
        log["锚点C_T18对应Z系数方向"] = {"系数": round(float(zc), 4), "为正": bool(zc > 0)}
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(os.path.join(OUT, "q4_coefficients.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "q4_model_spec.json"), "w", encoding="utf-8") as f:
    json.dump(spec, f, ensure_ascii=False, indent=2)
assert log["锚点C_T18对应Z系数方向"]["为正"]
# 规格自检：用 JSON 重算全数据概率与模型一致
for k in ["T18"]:
    s = spec[k]
    z = sum(f_["coef"] * (X[f_["name"]].values - f_["mean"]) / f_["std"] for f_ in s["features"]) + s["intercept"]
    p_json = 1 / (1 + np.exp(-z))
    m, _ = m1_fn(X, df[f"y_{k}"].values, groups)
    p_model = m.predict_proba(X)[:, 1]
    err_d = float(np.max(np.abs(p_json - p_model)))
    print(f"anchorD {k}: err={err_d:.6g}")
    log["锚点D_JSON规格复算误差"] = err_d
    assert err_d < 1e-4

for k in LABELS:
    np.save(os.path.join(OUT, f"q4_oof_m1_{k}.npy"), oof_m1[k])
with open(os.path.join(OUT, "q4_run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(log, ensure_ascii=False, indent=2, default=str)[:4500])
print("DONE")
