# -*- coding: utf-8 -*-
"""Q3 绘图：从 04_结果 主表/曲线重画论文可用图。运行: python q3_plot.py"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "04_结果")
FIG = os.path.join(ROOT, "03_可视化")
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300})
C1, C2, C_PT, C_TH = "#0072B2", "#D55E00", "#999999", "#CC0000"

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 图1 核心：候选因素时间比森林图（支持：BMI 之外因素均无显著增量效应）
scr = pd.read_csv(os.path.join(RES, "q3_screen.csv"))
eff = pd.read_csv(os.path.join(RES, "q3_effects_main.csv"))
bmi_row = eff[(eff["分布"] == "lognormal") & (eff["项"] == "alpha1_B0c")].iloc[0]
items = [("BMI（每 kg/m²，保留）", bmi_row["时间比TR"], np.exp(bmi_row["CI下"]), np.exp(bmi_row["CI上"]))]
lab_map = {"age_c": "年龄（每岁）", "height_c": "身高（每 cm）",
           "IVF": "IUI / IVF（vs 自然受孕）", "preg": "怀孕 2 次 / ≥3 次（vs 1 次）",
           "birth": "生产 1 次 / 2+ 次（vs 0 次）"}
for _, r in scr.iterrows():
    names = r["新增项"].split(";")
    est = [float(v) for v in str(r["估计"]).split(";")]
    lo = [float(v) for v in str(r["CI下"]).split(";")]
    hi = [float(v) for v in str(r["CI上"]).split(";")]
    labs = lab_map[r["候选"]].split(" / ")
    for k, c in enumerate(names):
        if max(abs(lo[k]), abs(hi[k])) > 5:   # 稀疏类别区间无效，截断标注
            items.append((f"{labs[min(k, len(labs)-1)]}（n 过少）", np.exp(est[k]), np.exp(max(lo[k], -2.0)), np.exp(min(hi[k], 2.0))))
        else:
            items.append((f"{labs[min(k, len(labs)-1)]}", np.exp(est[k]), np.exp(lo[k]), np.exp(hi[k])))
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ys = np.arange(len(items))[::-1]
for y, (lab, tr, lo, hi), in zip(ys, items):
    color = C1 if "保留" in lab else C_PT
    ax.plot([lo, hi], [y, y], color=color, lw=1.4)
    ax.plot([tr], [y], "o", color=color, ms=5)
    ax.text(3.4, y, lab, va="center", fontsize=7)
ax.axvline(1.0, color=C_TH, lw=1, ls=":")
ax.set(xscale="log", xlim=(0.5, 4), yticks=[],
       xlabel="达标时间比 TR（>1 表示达标更晚；横线=95%CI，log 尺度）",
       title="多因素对首次达标时间的影响：仅 BMI 的区间不含 1")
save(fig, "result_q3_effect_forest.png")

# 图2 核心：Q3 主方案两组达标曲线与推荐时点（与 Q2 一致）
TGRID = np.arange(11.0, 25.01, 0.5)
F = np.load(os.path.join(RES, "q3_grid_F.npy"))
order = pd.read_csv(os.path.join(RES, "q3_order.csv"))
cuts = json.load(open(os.path.join(RES, "q3_cuts.json")))["cuts"]
main = pd.read_csv(os.path.join(RES, "q3_groups_main.csv"))
fig, ax = plt.subplots(figsize=(6.3, 3.4))
for gi, (i, j) in enumerate(cuts):
    pbar = F[:, i:j].mean(axis=1)
    row = main.iloc[gi]
    cc = [C1, C2][gi]
    ax.plot(TGRID, pbar, color=cc, lw=1.6,
            label=f"{row['BMI区间']}（n={row['人数']}）：推荐 {row['推荐孕周']:.1f} 周")
    ax.plot([row["推荐孕周"]], [row["达标比例p(t*)"]], "o", color=cc, ms=6)
    ax.axvline(row["推荐孕周"], color=cc, lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color=C_TH, lw=1, ls=":")
ax.text(11.2, 0.952, "可靠度 0.95", color=C_TH, fontsize=7)
ax.set(xlabel="检测孕周（周）", ylabel="组内平均首次跨越概率", ylim=(0, 1.02))
ax.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "result_q3_group_timing.png")

# 图3 辅助：bootstrap 稳定性
stab = pd.read_csv(os.path.join(RES, "q3_stability_boot.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1, 1.2]})
bvals = stab.loc[stab["组"] == 1, "bounds"].astype(float)
axes[0].hist(bvals, bins=20, color=C1, alpha=0.8, edgecolor="white", linewidth=0.5)
axes[0].axvline(30.76, color=C_TH, lw=1.2)
axes[0].set(xlabel="分界点（kg/m²）", ylabel="频次", title="分界点稳定性（主解 30.8）")
rng = np.random.default_rng(1)
for gi, cc in zip([1, 2], [C1, C2]):
    v = stab.loc[stab["组"] == gi, "t*"]
    axes[1].scatter(np.full(len(v), gi) + rng.normal(0, 0.06, len(v)), v, s=6,
                    color=cc, alpha=0.35, linewidths=0)
    axes[1].boxplot([v], positions=[gi], widths=0.4, showfliers=False,
                    boxprops=dict(color=cc), medianprops=dict(color="#333333"),
                    whiskerprops=dict(color=cc), capprops=dict(color=cc))
axes[1].set(xlabel="组", ylabel="推荐孕周 t*（周）", title="t* 的 bootstrap 分布",
            xticks=[1, 2], xticklabels=["低 BMI 组", "高 BMI 组"])
save(fig, "process_q3_stability.png")
print("ALL DONE")
