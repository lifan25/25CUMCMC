# -*- coding: utf-8 -*-
"""Q4 绘图：PR 曲线、系数、标签-信号关系。运行: python q4_plot.py"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "04_结果")
FIG = os.path.join(ROOT, "03_可视化")
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300})
C1, C2, C3, C_PT, C_TH = "#0072B2", "#D55E00", "#009E73", "#999999", "#CC0000"

df = pd.read_csv(os.path.join(RES, "q4_features_labels.csv"))
base = pd.read_csv(os.path.join(RES, "q4_baselines.csv"))
LABELS = ["T13", "T18", "T21"]
ZCOL = {"T13": "13号染色体的Z值", "T18": "18号染色体的Z值", "T21": "21号染色体的Z值"}
CC = {"T13": C1, "T18": C2, "T21": C3}

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 图1 核心：三条染色体 PR 曲线（M1 折叠外预测）
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7))
for ax, k in zip(axes, LABELS):
    oof = np.load(os.path.join(RES, f"q4_oof_m1_{k}.npy"))
    y = df[f"y_{k}"].values
    pr, rc, _ = precision_recall_curve(y, oof)
    ap = average_precision_score(y, oof)
    ax.plot(rc, pr, color=CC[k], lw=1.5, label=f"M1 逻辑回归（AP={ap:.2f}）")
    ax.axhline(y.mean(), color=C_PT, lw=0.8, ls="--")
    ax.text(0.02, y.mean() + 0.02, f"随机水平 {y.mean():.3f}", fontsize=6, color=C_PT)
    r13 = base[(base["染色体"] == k) & (base["规则"].str.contains("教科书"))].iloc[0]
    ax.plot([r13["召回"]], [r13["精确率"]], "s", color=C_TH, ms=5)
    ax.annotate("|Z|>3 规则", (r13["召回"], r13["精确率"]), textcoords="offset points",
                xytext=(4, 6), fontsize=6, color=C_TH)
    ax.set(xlabel="召回率", ylabel="精确率", xlim=(0, 1), ylim=(0, 1), title=k)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
save(fig, "result_q4_pr_curves.png")

# 图2 核心：各染色体模型系数（前 8 特征，标准化）
coef = pd.read_csv(os.path.join(RES, "q4_coefficients.csv"))
fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=False)
for ax, k in zip(axes, LABELS):
    sub = coef[coef["染色体"] == k].head(8).iloc[::-1]
    colors = [C1 if v > 0 else C2 for v in sub["标准化系数"]]
    ax.barh(sub["特征"], sub["标准化系数"], color=colors, alpha=0.85, height=0.62)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set(xlabel="标准化系数", title=k)
    ax.tick_params(labelsize=6.5)
save(fig, "result_q4_coefficients.png")

# 图3 核心证据：Z 值按标签分布（解释 |Z|>3 规则为何失效）
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8))
rng = np.random.default_rng(1)
for ax, k in zip(axes, LABELS):
    z = df[ZCOL[k]].values; y = df[f"y_{k}"].values
    ax.scatter(np.zeros((y == 0).sum()) + rng.normal(0, 0.05, (y == 0).sum()), z[y == 0],
               s=4, c=C_PT, alpha=0.25, linewidths=0)
    ax.scatter(np.ones((y == 1).sum()) + rng.normal(0, 0.05, (y == 1).sum()), z[y == 1],
               s=14, c=C_TH, alpha=0.8, linewidths=0)
    ax.axhline(3, color=C_TH, lw=0.9, ls=":")
    ax.axhline(-3, color=C_TH, lw=0.9, ls=":")
    ax.set(xticks=[0, 1], xticklabels=["阴性", "阳性"], ylabel="Z 值", title=f"{k}（虚线=±3）")
save(fig, "raw_q4_z_by_label.png")
print("ALL DONE")
