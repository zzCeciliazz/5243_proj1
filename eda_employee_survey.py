import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kruskal, chi2_contingency
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Global Style
# ─────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 130,
})
PALETTE = "muted"
ACCENT  = "#4C72B0"
RED     = "#DD4949"
GREEN   = "#2CA02C"

# ─────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────
df = pd.read_csv("employee_survey_sleephours_final.csv")
OUT = "./"

# Cast categorical strings
cat_cols   = ["maritalstatus","joblevel","occupation","emptype","commutemode","edulevel","gender"]
likert_cols= ["wlb","workenv","workload","stress","jobsatisfaction"]
num_cols   = ["age","experience","physicalactivityhours","sleephours","commutedistance",
              "numcompanies","teamsize","numreports","traininghoursperyear",
              "work_schedule_variability","work_pace_pca"]

for c in cat_cols:
    df[c] = df[c].astype(str).str.strip()

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_output/")
os.makedirs(OUT, exist_ok=True)

# ════════════════════════════════════════════════════════════
# FIGURE 1 — Dataset Overview: Missing Values + Data Types
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 1 — Dataset Overview", fontsize=15, fontweight="bold", y=1.01)

# 1a: Missing value heatmap (column-wise %)
missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
if len(missing_pct) == 0:
    axes[0].text(0.5, 0.5, "No Missing Values\nin Dataset ✓",
                 ha="center", va="center", fontsize=14, color=GREEN,
                 transform=axes[0].transAxes)
    axes[0].set_axis_off()
else:
    missing_pct.plot(kind="barh", ax=axes[0], color=RED)
    axes[0].set_xlabel("Missing %")
axes[0].set_title("(a) Missing Values by Column")

# 1b: Data type distribution (pie)
dtype_counts = df.dtypes.apply(lambda x: x.name).value_counts()
dtype_labels = [f"{k}\n({v} cols)" for k, v in dtype_counts.items()]
axes[1].pie(dtype_counts, labels=dtype_labels, autopct="%1.0f%%",
            colors=sns.color_palette(PALETTE, len(dtype_counts)),
            startangle=140, textprops={"fontsize": 10})
axes[1].set_title("(b) Column Data Types")

plt.tight_layout()
plt.savefig(OUT + "fig1_overview.png", bbox_inches="tight")
plt.close()
print("Saved fig1_overview.png")

# ════════════════════════════════════════════════════════════
# FIGURE 2 — Univariate: Key Numerical Distributions
# ════════════════════════════════════════════════════════════
key_nums = ["age","sleephours","physicalactivityhours","experience",
            "traininghoursperyear","commutedistance"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Figure 2 — Distributions of Key Numerical Variables", fontsize=15, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(key_nums):
    ax = axes[i]
    data = df[col].dropna()
    sns.histplot(data, kde=True, ax=ax, color=ACCENT, edgecolor="white", linewidth=0.4)
    ax.axvline(data.mean(),   color=RED,   linestyle="--", linewidth=1.2, label=f"Mean={data.mean():.1f}")
    ax.axvline(data.median(), color=GREEN, linestyle=":",  linewidth=1.2, label=f"Median={data.median():.1f}")
    sk = stats.skew(data)
    ax.set_title(f"{col}\n(skew={sk:.2f})")
    ax.set_xlabel("")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT + "fig2_num_distributions.png", bbox_inches="tight")
plt.close()
print("Saved fig2_num_distributions.png")

# ════════════════════════════════════════════════════════════
# FIGURE 3 — Univariate: Categorical & Likert Variables
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Figure 3 — Categorical & Likert Variable Distributions", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# Categorical
cat_plot = ["occupation","joblevel","emptype","edulevel","commutemode"]
for i, col in enumerate(cat_plot):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    vc = df[col].value_counts()
    vc.plot(kind="bar", ax=ax, color=sns.color_palette(PALETTE, len(vc)), edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=7)

# Likert (bottom row)
ax_likert = fig.add_subplot(gs[2, :])
likert_pct = df[likert_cols].apply(lambda x: x.value_counts(normalize=True).sort_index() * 100)
likert_pct = likert_pct.T
likert_pct.columns = [f"Level {c}" for c in likert_pct.columns]
likert_pct.plot(kind="bar", ax=ax_likert,
                color=sns.color_palette("Blues_d", 5), edgecolor="white", width=0.7)
ax_likert.set_title("Likert Scales — Response Distribution (%)")
ax_likert.set_xticklabels(ax_likert.get_xticklabels(), rotation=0, fontsize=10)
ax_likert.set_ylabel("Percentage (%)")
ax_likert.legend(title="Score", fontsize=8, title_fontsize=8)

plt.savefig(OUT + "fig3_categorical.png", bbox_inches="tight")
plt.close()
print("Saved fig3_categorical.png")

# ════════════════════════════════════════════════════════════
# FIGURE 4 — Correlation Heatmap (Pearson + Spearman)
# ════════════════════════════════════════════════════════════
corr_cols = num_cols + likert_cols + ["haveot"]
corr_df   = df[corr_cols].select_dtypes(include=np.number)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle("Figure 4 — Correlation Matrices", fontsize=15, fontweight="bold")

for ax, method, title in zip(axes, ["pearson","spearman"],
                              ["(a) Pearson Correlation (linear)","(b) Spearman Correlation (rank)"]):
    corr = corr_df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size":7},
                linewidths=0.3, square=False)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig(OUT + "fig4_correlation.png", bbox_inches="tight")
plt.close()
print("Saved fig4_correlation.png")

# ════════════════════════════════════════════════════════════
# FIGURE 5 — SleepHours Deep Dive
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Figure 5 — SleepHours: Deep-Dive Analysis", fontsize=15, fontweight="bold")

sleep = df["sleephours"]

# 5a: Boxplot by stress
ax = axes[0,0]
df.boxplot(column="sleephours", by="stress", ax=ax,
           patch_artist=True,
           boxprops=dict(facecolor="#AEC6CF"),
           medianprops=dict(color=RED, linewidth=2))
ax.set_title("(a) SleepHours by Stress Level")
ax.set_xlabel("Stress Level")
ax.set_ylabel("Sleep Hours")
plt.sca(ax); plt.title("(a) SleepHours by Stress Level")

# 5b: Boxplot by joblevel
ax = axes[0,1]
order = ["Intern/Fresher","Junior","Mid","Senior","Manager","Director/VP","C-Suite"]
order = [o for o in order if o in df["joblevel"].unique()]
data_grouped = [df[df["joblevel"]==j]["sleephours"].dropna() for j in order]
bp = ax.boxplot(data_grouped, labels=order, patch_artist=True,
                medianprops=dict(color=RED, linewidth=2))
colors = sns.color_palette("muted", len(order))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_title("(b) SleepHours by Job Level")
ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Sleep Hours")

# 5c: SleepHours vs workload scatter (with regression)
ax = axes[0,2]
jitter = np.random.uniform(-0.15, 0.15, len(df))
ax.scatter(df["workload"]+jitter, df["sleephours"], alpha=0.15, s=12, color=ACCENT)
for wl in sorted(df["workload"].unique()):
    g = df[df["workload"]==wl]["sleephours"]
    ax.plot(wl, g.mean(), "o", color=RED, markersize=7, zorder=5)
m, b, r, p, _ = stats.linregress(df["workload"].dropna(), df["sleephours"].dropna())
x_line = np.linspace(1,5,100)
ax.plot(x_line, m*x_line+b, "--", color=RED, linewidth=1.5, label=f"r={r:.2f}, p={p:.3f}")
ax.set_title("(c) SleepHours vs Workload")
ax.set_xlabel("Workload"); ax.set_ylabel("Sleep Hours")
ax.legend(fontsize=9)

# 5d: SleepHours vs physicalactivity
ax = axes[1,0]
ax.scatter(df["physicalactivityhours"], df["sleephours"], alpha=0.2, s=12, color=GREEN)
m,b,r,p,_ = stats.linregress(df["physicalactivityhours"].dropna(), df["sleephours"].dropna())
x_l = np.linspace(df["physicalactivityhours"].min(), df["physicalactivityhours"].max(), 100)
ax.plot(x_l, m*x_l+b, "--", color=RED, lw=1.5, label=f"r={r:.2f}, p={p:.3f}")
ax.set_title("(d) SleepHours vs Physical Activity")
ax.set_xlabel("Physical Activity (hrs)"); ax.set_ylabel("Sleep Hours")
ax.legend(fontsize=9)

# 5e: Mean SleepHours by occupation
ax = axes[1,1]
occ_sleep = df.groupby("occupation")["sleephours"].mean().sort_values()
occ_sleep.plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(occ_sleep)))
ax.axvline(sleep.mean(), color=RED, linestyle="--", lw=1.2, label=f"Overall mean={sleep.mean():.2f}")
ax.set_title("(e) Mean SleepHours by Occupation")
ax.set_xlabel("Avg Sleep Hours")
ax.legend(fontsize=8)

# 5f: Violin — SleepHours by WLB
ax = axes[1,2]
sns.violinplot(data=df, x="wlb", y="sleephours", ax=ax, palette="muted", inner="box")
ax.set_title("(f) SleepHours by Work-Life Balance")
ax.set_xlabel("WLB Score"); ax.set_ylabel("Sleep Hours")

plt.tight_layout()
plt.savefig(OUT + "fig5_sleep_deepdive.png", bbox_inches="tight")
plt.close()
print("Saved fig5_sleep_deepdive.png")

# ════════════════════════════════════════════════════════════
# FIGURE 6 — Stress & Wellbeing Ecosystem
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Figure 6 — Stress & Employee Wellbeing Ecosystem", fontsize=15, fontweight="bold")

# 6a: Stress vs Workload heatmap
ax = axes[0,0]
ct = pd.crosstab(df["stress"], df["workload"], normalize="all") * 100
sns.heatmap(ct, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
            annot_kws={"size":8}, linewidths=0.3)
ax.set_title("(a) Stress × Workload Joint Distribution (%)")
ax.set_xlabel("Workload"); ax.set_ylabel("Stress")

# 6b: Stress by occupation (mean)
ax = axes[0,1]
occ_stress = df.groupby("occupation")["stress"].mean().sort_values(ascending=False)
occ_stress.plot(kind="bar", ax=ax, color=sns.color_palette("Reds_r", len(occ_stress)), edgecolor="white")
ax.set_title("(b) Mean Stress by Occupation")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Mean Stress Score")

# 6c: WLB vs JobSatisfaction (scatter + regression)
ax = axes[0,2]
jitter_x = np.random.uniform(-0.2,0.2,len(df))
jitter_y = np.random.uniform(-0.2,0.2,len(df))
sc = ax.scatter(df["wlb"]+jitter_x, df["jobsatisfaction"]+jitter_y, alpha=0.15, s=10, c=df["stress"],
                cmap="RdYlGn_r", vmin=1, vmax=5)
plt.colorbar(sc, ax=ax, label="Stress Level", pad=0.01)
m,b,r,p,_ = stats.linregress(df["wlb"], df["jobsatisfaction"])
x_l = np.linspace(1,5,100)
ax.plot(x_l, m*x_l+b, "--", color=ACCENT, lw=1.5, label=f"r={r:.2f}")
ax.set_title("(c) WLB vs Job Satisfaction\n(colored by stress)")
ax.set_xlabel("WLB"); ax.set_ylabel("Job Satisfaction")
ax.legend(fontsize=9)

# 6d: OT vs Stress (stacked bar)
ax = axes[1,0]
ct_ot = pd.crosstab(df["stress"], df["haveot"], normalize="index") * 100
ct_ot.columns = ["No OT","Has OT"]
ct_ot.plot(kind="bar", stacked=True, ax=ax,
           color=["#AEC6CF","#FF7F7F"], edgecolor="white")
ax.set_title("(d) Overtime Distribution by Stress Level")
ax.set_xlabel("Stress Level"); ax.set_ylabel("Proportion (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(fontsize=9)

# 6e: WorkEnv vs JobSatisfaction violin
ax = axes[1,1]
sns.violinplot(data=df, x="workenv", y="jobsatisfaction", ax=ax,
               palette="Blues", inner="quartile")
ax.set_title("(e) Job Satisfaction by Work Environment")
ax.set_xlabel("Work Environment Score"); ax.set_ylabel("Job Satisfaction")

# 6f: Stress, Workload, WLB, Sleep heatmap (group mean by joblevel)
ax = axes[1,2]
jl_order = [j for j in ["Intern/Fresher","Junior","Mid","Senior","Manager","Director/VP","C-Suite"]
            if j in df["joblevel"].unique()]
hm_data = df.groupby("joblevel")[["stress","workload","wlb","sleephours"]].mean().loc[jl_order]
hm_norm = (hm_data - hm_data.min()) / (hm_data.max() - hm_data.min())
sns.heatmap(hm_norm, ax=ax, cmap="RdYlGn_r", annot=hm_data, fmt=".2f",
            annot_kws={"size":8}, linewidths=0.4)
ax.set_title("(f) Wellbeing Profile by Job Level\n(color=normalized, value=raw mean)")
ax.set_ylabel("Job Level"); ax.set_xticklabels(ax.get_xticklabels(), rotation=20)

plt.tight_layout()
plt.savefig(OUT + "fig6_stress_wellbeing.png", bbox_inches="tight")
plt.close()
print("Saved fig6_stress_wellbeing.png")

# ════════════════════════════════════════════════════════════
# FIGURE 7 — Work Behaviour & Demographics
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Figure 7 — Work Behaviour & Demographic Patterns", fontsize=15, fontweight="bold")

# 7a: Age distribution by emptype
ax = axes[0,0]
for emp, color in zip(df["emptype"].unique(), sns.color_palette("muted")):
    subset = df[df["emptype"]==emp]["age"].dropna()
    subset.plot.kde(ax=ax, label=emp, color=color, linewidth=1.8)
ax.set_title("(a) Age Distribution by Employment Type")
ax.set_xlabel("Age"); ax.set_ylabel("Density")
ax.legend(fontsize=8)

# 7b: Training hours by joblevel
ax = axes[0,1]
sns.boxplot(data=df, x="joblevel", y="traininghoursperyear", ax=ax,
            order=jl_order, palette="muted")
ax.set_title("(b) Training Hours by Job Level")
ax.set_xticklabels(jl_order, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Training Hrs/Year")

# 7c: Commute mode vs stress (grouped bar)
ax = axes[0,2]
ct = df.groupby(["commutemode","stress"]).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind="bar", ax=ax, colormap="RdYlGn_r", edgecolor="white", width=0.75)
ax.set_title("(c) Stress Distribution by Commute Mode")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Proportion (%)")
ax.legend(title="Stress", fontsize=7, title_fontsize=7)

# 7d: NumReports vs TeamSize scatter
ax = axes[1,0]
sc = ax.scatter(df["teamsize"], df["numreports"], alpha=0.2, s=15,
                c=df["joblevel"].map({j:i for i,j in enumerate(jl_order)}),
                cmap="viridis")
ax.set_title("(d) Team Size vs Num Reports")
ax.set_xlabel("Team Size"); ax.set_ylabel("Num Reports")

# 7e: Work pace PCA distribution by emptype
ax = axes[1,1]
sns.boxplot(data=df, x="emptype", y="work_pace_pca", ax=ax, palette="Set2")
ax.set_title("(e) Work Pace (PCA) by Employment Type")
ax.set_xlabel("Employment Type"); ax.set_ylabel("Work Pace (PCA Score)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

# 7f: Pair plot substitute — key variable correlations scatter matrix
ax = axes[1,2]
pair_vars = ["stress","sleephours","wlb","physicalactivityhours","jobsatisfaction"]
corr_sub = df[pair_vars].corr(method="spearman")
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, ax=ax, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            annot_kws={"size":10}, linewidths=0.5, square=True,
            mask=mask)
ax.set_title("(f) Spearman Corr: Wellbeing Variables")

plt.tight_layout()
plt.savefig(OUT + "fig7_demographics_behaviour.png", bbox_inches="tight")
plt.close()
print("Saved fig7_demographics_behaviour.png")

# ════════════════════════════════════════════════════════════
# FIGURE 8 — Statistical Tests Summary
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figure 8 — Statistical Test Results", fontsize=15, fontweight="bold")

# 8a: Kruskal-Wallis: SleepHours across stress groups
ax = axes[0]
groups = [df[df["stress"]==s]["sleephours"].dropna() for s in sorted(df["stress"].unique())]
kw_stat, kw_p = kruskal(*groups)

# Mean + CI per stress group
means, cis, labels_ = [], [], []
for s in sorted(df["stress"].unique()):
    g = df[df["stress"]==s]["sleephours"].dropna()
    mean = g.mean()
    ci   = 1.96 * g.std() / np.sqrt(len(g))
    means.append(mean); cis.append(ci); labels_.append(f"Stress {s}\n(n={len(g)})")

colors_ = sns.color_palette("RdYlGn_r", 5)
bars = ax.bar(labels_, means, yerr=cis, capsize=5,
              color=colors_, edgecolor="white", error_kw={"linewidth":1.5})
ax.set_title(f"(a) Mean SleepHours by Stress Level\nKruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f}")
ax.set_ylabel("Avg Sleep Hours")
ax.axhline(df["sleephours"].mean(), color="gray", linestyle="--", lw=1, label="Overall mean")
ax.legend(fontsize=9)

# 8b: Chi-square: OT × JobLevel
ax = axes[1]
ct_chi = pd.crosstab(df["joblevel"], df["haveot"])
chi2, p_chi, dof, _ = chi2_contingency(ct_chi)
ct_pct = ct_chi.div(ct_chi.sum(axis=1), axis=0) * 100
ct_pct = ct_pct.reindex(jl_order)
ct_pct.columns = ["No OT","Has OT"]
ct_pct.plot(kind="barh", ax=ax, color=["#AEC6CF","#FF7F7F"], edgecolor="white", width=0.65)
ax.set_title(f"(b) Overtime Prevalence by Job Level\nχ² = {chi2:.1f}, p = {p_chi:.4f}, df={dof}")
ax.set_xlabel("Proportion (%)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT + "fig8_stat_tests.png", bbox_inches="tight")
plt.close()
print("Saved fig8_stat_tests.png")

# ════════════════════════════════════════════════════════════
# Print EDA Summary Stats
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EDA SUMMARY STATISTICS")
print("="*60)
print(f"Dataset: {df.shape[0]} employees × {df.shape[1]} variables")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nSleepHours: mean={df['sleephours'].mean():.2f}, "
      f"std={df['sleephours'].std():.2f}, "
      f"skew={stats.skew(df['sleephours'].dropna()):.3f}")
print(f"\nTop 3 correlates with sleephours (Spearman):")
sleep_corr = df[corr_cols].corr(method="spearman")["sleephours"].drop("sleephours").abs().sort_values(ascending=False)
print(sleep_corr.head(3).to_string())
print(f"\nKruskal-Wallis (SleepHours across Stress): H={kw_stat:.2f}, p={kw_p:.4f}")
print(f"Chi-square (OT × JobLevel): χ²={chi2:.1f}, p={p_chi:.4f}")
print("\nAll figures saved to /mnt/user-data/outputs/")
