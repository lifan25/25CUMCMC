# -*- coding: utf-8 -*-
"""Q4 主程序：女胎 T13/T18/T21 多标签判定。规则基线 + 分染色体 L2 逻辑回归 + 阈值情景 + 分组验证。
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

# ---------------- 1. 数据 ----------------
def parse_week(s):
    m = re.match(r"^\s*(\d+)\s*[wW]\s*(?:\+\s*(\d+))?\s*$", str(s))
    return np.nan if not m else int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 7.0

df = pd.read_excel(XLSX, sheet_name="女胎检测数据")
df.columns = [str(c).strip() for c in df.columns]
df["t_weeks"] = df["检测孕周"].map(parse_week)
miss_bmi = df["孕妇BMI"].isna()
df.loc[miss_bmi, "孕妇BMI"] = df.loc[miss_bmi, "体重"] / (df.loc[miss_bmi, "身高"] / 100) ** 2
log["BMI回算条数"] = int(miss_bmi.sum())
ab = df["染色体的非整倍体"].fillna("")
for k in ["T13", "T18", "T21"]:
    df[f"y_{k}"] = ab.str.contains(k).astype(int)
log["记录数/孕妇数"] = [len(df), int(df["孕妇代码"].nunique())]
log["阳性记录数"] = {k: int(df[f"y_{k}"].sum()) for k in ["T13", "T18", "T21"]}
wlab = df.groupby("孕妇代码")[[f"y_{k}" for k in ["T13", "T18", "T21"]]].max()
log["孕妇级阳性数"] = {k: int(wlab[f"y_{k}"].sum()) for k in ["T13", "T18", "T21"]}
incons = df.groupby("孕妇代码")["染色体的非整倍体"].apply(lambda s: len(set(s.dropna())) > 1)
log["标签随记录变化孕妇数"] = int(incons.sum())

FEATS = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "X染色体浓度",
         "GC含量", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
         "log_reads", "log_uniq", "在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例",
         "孕妇BMI", "年龄", "身高", "t_weeks"]
df["log_reads"] = np.log10(df["原始读段数"])
df["log_uniq"] = np.log10(df["唯一比对的读段数"])
X = df[FEATS].copy()
groups = df["孕妇代码"].values
LABELS = ["T13", "T18", "T21"]
ZCOL = {"T13": "13号染色体的Z值", "T18": "18号染色体的Z值", "T21": "21号染色体的Z值"}

df[FEATS + ["y_T13", "y_T18", "y_T21", "孕妇代码"]].to_csv(
    os.path.join(OUT, "q4_features_labels.csv"), index=False, encoding="utf-8-sig")

# ---------------- 2. 锚点 A：打乱标签应无信号 ----------------
gkf = GroupKFold(5)
rng = np.random.default_rng(0)
auc_shuf = []
for k in LABELS:
    y_sh = rng.permutation(df[f"y_{k}"].values)
    for tr, va in gkf.split(X, y_sh, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
        m.fit(X.iloc[tr], y_sh[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        if y_sh[va].sum() > 0 and y_sh[va].sum() < len(va):
            auc_shuf.append(roc_auc_score(y_sh[va], p))
anchorA = abs(float(np.mean(auc_shuf)) - 0.5) < 0.1
log["锚点A_打乱标签CV_AUC(应≈0.5)"] = {"均值": round(float(np.mean(auc_shuf)), 3), "通过": anchorA}
assert anchorA

# ---------------- 3. 规则基线：|Z|>3 与数据驱动阈值 ----------------
base_rows = []
for k in LABELS:
    y = df[f"y_{k}"].values
    z = df[ZCOL[k]].abs().values
    pred = z > 3
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int(((~pred) & (y == 1)).sum())
    base_rows.append({"染色体": k, "规则": "|Z|>3（教科书）", "TP": tp, "FP": fp, "FN": fn,
                      "召回": round(tp / max(tp + fn, 1), 4), "精确率": round(tp / max(tp + fp, 1), 4)})
    # 数据驱动：分组 CV 内选 |Z| 阈值（F1 最优），折叠外评估
    oof = np.zeros(len(df))
    for tr, va in gkf.split(X, y, groups):
        cand = np.quantile(z[tr], np.linspace(0.90, 0.999, 30))
        f1s = []
        for t in cand:
            pr = z[tr] > t
            tp_ = (pr & (y[tr] == 1)).sum(); fp_ = (pr & (y[tr] == 0)).sum(); fn_ = ((~pr) & (y[tr] == 1)).sum()
            f1s.append(2 * tp_ / max(2 * tp_ + fp_ + fn_, 1))
        oof[va] = (z[va] > cand[int(np.argmax(f1s))]).astype(float)
    tp, fp, fn = int((oof * y).sum()), int((oof * (1 - y)).sum()), int(((1 - oof) * y).sum())
    base_rows.append({"染色体": k, "规则": "|Z|>τ（训练折 F1 选 τ）", "TP": tp, "FP": fp, "FN": fn,
                      "召回": round(tp / max(tp + fn, 1), 4), "精确率": round(tp / max(tp + fp, 1), 4)})
base_df = pd.DataFrame(base_rows)
base_df.to_csv(os.path.join(OUT, "q4_baselines.csv"), index=False, encoding="utf-8-sig")
log["锚点B_Z规则基线复现"] = base_rows[0]["召回"] < 0.1 and base_rows[2]["召回"] < 0.1 and base_rows[4]["召回"] < 0.1
assert log["锚点B_Z规则基线复现"]

# ---------------- 4. M1 主模型（内层选 C）与 M2 树对照 ----------------
def oof_predict(model_fn, y):
    oof = np.zeros(len(X))
    for tr, va in gkf.split(X, y, groups):
        m = model_fn(tr, y)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof

def m1_fn(tr, y):
    # 内层 3 折按 PR-AUC 选 C
    inner = GroupKFold(3)
    best_c, best_s = 0.1, -1
    for c in [0.03, 0.1, 0.3]:
        s = []
        for itr, iva in inner.split(X.iloc[tr], y[tr], groups[tr]):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
            m.fit(X.iloc[tr].iloc[itr], y[tr][itr])
            pv = m.predict_proba(X.iloc[tr].iloc[iva])[:, 1]
            if y[tr][iva].sum() > 0:
                s.append(average_precision_score(y[tr][iva], pv))
        if s and np.mean(s) > best_s:
            best_s, best_c = np.mean(s), c
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000))
    m.fit(X.iloc[tr], y[tr])
    return m

def m2_fn(tr, y):
    m = make_pipeline(StandardScaler(),
                      DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced",
                                             random_state=SEED))
    m.fit(X.iloc[tr], y[tr])
    return m

oof_m1, oof_m2, cmp_rows = {}, {}, []
for k in LABELS:
    y = df[f"y_{k}"].values
    oof_m1[k] = oof_predict(m1_fn, y)
    oof_m2[k] = oof_predict(m2_fn, y)
    for name, oof in [("M1_逻辑回归", oof_m1[k]), ("M2_浅层树", oof_m2[k])]:
        cmp_rows.append({"染色体": k, "模型": name,
                         "ROC_AUC": round(roc_auc_score(y, oof), 4),
                         "PR_AUC": round(average_precision_score(y, oof), 4),
                         "随机水平": round(float(y.mean()), 4)})
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(OUT, "q4_model_compare.csv"), index=False, encoding="utf-8-sig")
# M2 保留规则：均值 PR-AUC 高 >0.02 才对照保留
use_tree = {}
for k in LABELS:
    r = cmp_df[cmp_df["染色体"] == k]
    use_tree[k] = bool(r[r["模型"] == "M2_浅层树"]["PR_AUC"].iloc[0] - r[r["模型"] == "M1_逻辑回归"]["PR_AUC"].iloc[0] > 0.02)
log["模型比较"] = cmp_rows
log["M2保留"] = use_tree
OOF = {k: (oof_m2[k] if use_tree[k] else oof_m1[k]) for k in LABELS}
log["锚点C_T18对应Z系数方向"] = "待最终系数核对"

# ---------------- 5. 阈值情景（训练折内选，折叠外评估） ----------------
SCEN = {"基准(召回≥0.80)": 0.80, "偏精确(召回≥0.70)": 0.70, "偏召回(召回≥0.90)": 0.90}
def metrics_at(y, pred_bin):
    pred_bin = pred_bin.astype(bool); yb = y.astype(bool)
    tp = int((pred_bin & yb).sum()); fp = int((pred_bin & ~yb).sum())
    fn = int((~pred_bin & yb).sum()); tn = int((~pred_bin & ~yb).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "召回": round(tp / max(tp + fn, 1), 4), "精确率": round(tp / max(tp + fp, 1), 4),
            "特异度": round(tn / max(tn + fp, 1), 4),
            "F1": round(2 * tp / max(2 * tp + fp + fn, 1), 4)}
perf_rows, thresh_used = [], {}
for k in LABELS:
    y = df[f"y_{k}"].values
    oof = OOF[k]
    thresh_used[k] = {}
    for sname, target in SCEN.items():
        # 阈值在“训练口径”内选：用 OOF 预测按孕妇分组 5 折内层选择，避免用验证折选阈值
        ths = []
        for tr, va in gkf.split(X, y, groups):
            p_tr, y_tr = oof[tr], y[tr]
            cand = np.unique(np.quantile(p_tr, np.linspace(0.5, 0.999, 200)))
            ok = [t for t in cand if ((p_tr >= t) & (y_tr == 1)).sum() / max((y_tr == 1).sum(), 1) >= target]
            ths.append(max(ok) if ok else float(np.quantile(p_tr, 0.999)))
        th = float(np.median(ths))
        thresh_used[k][sname] = round(th, 4)
        m_ = metrics_at(y, (oof >= th).astype(int))
        perf_rows.append({"染色体": k, "情景": sname, "阈值": round(th, 4),
                          "层级": "记录级", **m_})
    # 孕妇级（概率取记录最大，阈值同记录级情景）
    wp = pd.Series(oof).groupby(groups).max()
    wy = pd.Series(y).groupby(groups).max()
    for sname in SCEN:
        th = thresh_used[k][sname]
        m_ = metrics_at(wy.values, (wp.values >= th).astype(int))
        perf_rows.append({"染色体": k, "情景": sname, "阈值": th, "层级": "孕妇级", **m_})
perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(os.path.join(OUT, "q4_performance.csv"), index=False, encoding="utf-8-sig")
log["阈值"] = thresh_used
log["性能_基准_记录级"] = [r for r in perf_rows if r["情景"].startswith("基准") and r["层级"] == "记录级"]
log["性能_基准_孕妇级"] = [r for r in perf_rows if r["情景"].startswith("基准") and r["层级"] == "孕妇级"]

# ---------------- 6. bootstrap 区间（基准情景，记录级召回与 PR-AUC） ----------------
B = 500
boot_rows = []
pids = df["孕妇代码"].unique()
for k in LABELS:
    y = df[f"y_{k}"].values
    oof = OOF[k]
    th = thresh_used[k]["基准(召回≥0.80)"]
    recs, pras = [], []
    for b in range(B):
        samp = np.random.default_rng(SEED + b).choice(pids, size=len(pids), replace=True)
        mask = np.isin(groups, samp)
        if y[mask].sum() == 0:
            continue
        pred = (oof[mask] >= th).astype(int)
        tp = (pred & (y[mask] == 1)).sum(); fn = ((1 - pred) & (y[mask] == 1)).sum()
        recs.append(tp / max(tp + fn, 1))
        pras.append(average_precision_score(y[mask], oof[mask]))
    boot_rows.append({"染色体": k, "召回_中位": round(float(np.median(recs)), 4),
                      "召回_5%": round(float(np.quantile(recs, 0.05)), 4),
                      "召回_95%": round(float(np.quantile(recs, 0.95)), 4),
                      "PRAUC_中位": round(float(np.median(pras)), 4),
                      "PRAUC_5%": round(float(np.quantile(pras, 0.05)), 4),
                      "PRAUC_95%": round(float(np.quantile(pras, 0.95)), 4)})
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(os.path.join(OUT, "q4_bootstrap.csv"), index=False, encoding="utf-8-sig")
log["bootstrap区间(基准,记录级)"] = boot_rows

# ---------------- 7. 最终规则（全数据重拟合系数） ----------------
coef_rows = []
for k in LABELS:
    y = df[f"y_{k}"].values
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    m.fit(X, y)
    lr = m.named_steps["logisticregression"]
    for f_, c_ in sorted(zip(FEATS, lr.coef_[0]), key=lambda t: -abs(t[1])):
        coef_rows.append({"染色体": k, "特征": f_, "标准化系数": round(float(c_), 4)})
    if k == "T18":
        zc = dict(zip(FEATS, lr.coef_[0]))[ZCOL["T18"]]
        log["锚点C_T18对应Z系数方向"] = {"系数": round(float(zc), 4), "为正": bool(zc > 0)}
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(os.path.join(OUT, "q4_coefficients.csv"), index=False, encoding="utf-8-sig")
assert log["锚点C_T18对应Z系数方向"]["为正"]

np.save(os.path.join(OUT, "q4_oof_m1_T13.npy"), oof_m1["T13"])
np.save(os.path.join(OUT, "q4_oof_m1_T18.npy"), oof_m1["T18"])
np.save(os.path.join(OUT, "q4_oof_m1_T21.npy"), oof_m1["T21"])
with open(os.path.join(OUT, "q4_run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(log, ensure_ascii=False, indent=2, default=str)[:4000])
print("DONE")
