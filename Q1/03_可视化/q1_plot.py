# -*- coding: utf-8 -*-
"""Q1 绘图：从 04_结果 主表重画论文可用图。运行: python q1_plot.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "04_结果")
FIG = os.path.join(ROOT, "03_可视化")
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300})
C_MAIN, C_CMP, C_PT, C_TH = "#0072B2", "#D55E00", "#999999", "#CC0000"

ev = pd.read_csv(os.path.join(RES, "q1_blood_events.csv"))
curve = pd.read_csv(os.path.join(RES, "q1_effect_curve.csv"))
dg = pd.read_csv(os.path.join(RES, "q1_diagnostics.csv"))
rep = pd.read_csv(os.path.join(RES, "q1_technical_replicates.csv"))
chosen = dg["model"].iloc[0]

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 图1 结果图：孕周-浓度关系与模型曲线（支持：浓度随孕周上升，M0/M1 几乎重合）
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ax.scatter(ev["t_weeks"], ev["Y"], s=6, c=C_PT, alpha=0.35, linewidths=0, label="采血级观测")
ax.fill_between(curve["t_weeks"], curve["M0_lo"], curve["M0_hi"], color=C_MAIN, alpha=0.18, linewidths=0)
ax.plot(curve["t_weeks"], curve["M0_pred"], color=C_MAIN, lw=1.6, label="M0（LMM，采用）边际均值 95%CI")
ax.plot(curve["t_weeks"], curve["M1_pred"], color=C_CMP, lw=1.2, ls="--", label="M1（自然样条混合模型）对照")
ax.axhline(0.04, color=C_TH, lw=1, ls=":")
ax.text(ev["t_weeks"].max() - 0.2, 0.0415, "4% 阈值", color=C_TH, ha="right", fontsize=7)
ax.set(xlabel="检测孕周（周）", ylabel="Y 染色体浓度（比例）", xlim=(10.5, 25.5))
ax.legend(frameon=False, loc="upper left", fontsize=7)
save(fig, "result_q1_week_curve.png")

# 图2 结果图：模型调整后的基线 BMI 效应（部分残差图，控制孕周/ΔB/读段数；支持：BMI 独立负效应）
eff = pd.read_csv(os.path.join(RES, "q1_effects_main.csv"))
b2row = eff[(eff["模型"] == "M0") & (eff["项"] == "B0_c")].iloc[0]
b2, b2lo, b2hi = b2row["估计"], b2row["CI下"], b2row["CI上"]
B0_mean = (ev.groupby("pid").first()["B0"]).mean()
dg["B0_c"] = dg["B0"] - B0_mean
dg["part_resid"] = b2 * dg["B0_c"] + dg["resid"]
fig, ax = plt.subplots(figsize=(6.3, 3.2))
ax.scatter(dg["B0"], dg["part_resid"], s=6, c=C_PT, alpha=0.35, linewidths=0, label="部分残差（控制孕周等项）")
xg = np.linspace(dg["B0"].min(), dg["B0"].max(), 100)
ax.fill_between(xg, b2lo * (xg - B0_mean), b2hi * (xg - B0_mean), color=C_MAIN, alpha=0.18, linewidths=0)
ax.plot(xg, b2 * (xg - B0_mean), color=C_MAIN, lw=1.6,
        label=f"调整后 BMI 效应 β2={b2:.5f}（95%CI {b2lo:.5f}~{b2hi:.5f}）")
ax.set(xlabel="基线 BMI（kg/m²）", ylabel="BMI 分量 + 残差（比例）")
ax.legend(frameon=False, loc="upper right", fontsize=7)
save(fig, "result_q1_bmi_adjusted.png")

# 图3 过程图：残差诊断（支持：高斯假设可接受，轻度右偏）
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
sps.probplot(dg["resid"], dist="norm", plot=axes[0])
axes[0].get_lines()[0].set(ms=3, color=C_PT, alpha=0.5)
axes[0].get_lines()[1].set(color=C_MAIN, lw=1.2)
axes[0].set_title("残差正态 QQ", fontsize=8)
axes[0].set_xlabel("理论分位数"); axes[0].set_ylabel("样本分位数")
axes[1].scatter(dg["fitted"], dg["resid"], s=6, c=C_PT, alpha=0.35, linewidths=0)
axes[1].axhline(0, color=C_MAIN, lw=1)
lo = lowess(dg["resid"], dg["fitted"], frac=0.3, return_sorted=True)
axes[1].plot(lo[:, 0], lo[:, 1], color=C_CMP, lw=1.4)
axes[1].set(xlabel="拟合值（比例）", ylabel="残差（比例）", title="残差-拟合值")
save(fig, "process_q1_diagnostics.png")

# 图4 原始数据图：事件内偏差分布（支持：单次测序误差约0.61个百分点，相对4%不可忽略）
devs = []
for s in rep["Y_values"]:
    ys = np.array([float(v) for v in str(s).split(";")])
    devs.extend((ys - ys.mean()).tolist())
devs = np.array(devs)
ssw = float((devs ** 2).sum()); dfw = int(sum(int(str(s).count(";")) for s in rep["Y_values"]))
s_hat = np.sqrt(ssw / dfw)
fig, ax = plt.subplots(figsize=(6.3, 3.0))
ax.hist(devs, bins=18, color=C_MAIN, alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="#333333", lw=1)
ax.set(xlabel="同次采血重复检测相对事件均值的偏差（比例）", ylabel="频数",
       title=f"事件内偏差（n={len(devs)}）：合并方差 ŝ=√(SSW/{dfw})={s_hat:.4f}，为 4% 阈值的 {s_hat/0.04:.1%}")
save(fig, "raw_q1_replicate_diffs.png")
print("ALL DONE")
