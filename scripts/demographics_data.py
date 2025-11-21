
#%%
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, iqr, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# -- Helpers
def _mw(a, b, alternative="two-sided"):
    a = pd.Series(a).dropna().values
    b = pd.Series(b).dropna().values
    if len(a) == 0 or len(b) == 0:
        return None
    res = mannwhitneyu(a, b, alternative=alternative)
    u1 = res.statistic
    u_min = u1 if u1 < (len(a) * len(b) - u1) else (len(a) * len(b) - u1)
    return mannwhitneyu(a, b, alternative=alternative)._replace(statistic=u_min)


def _fmt_p(p):
    if p is None:
        return "NA"
    if p < 1e-3:
        return f"{p:.3e}"
    return f"{p:.3f}"


def _fmt_stat(name, stat, p):
    if stat is None or p is None:
        return f"{name} = NA, p = NA"
    if isinstance(stat, int | np.integer) or float(stat).is_integer():
        return f"{name} = {stat:.0f}, p = {_fmt_p(p)}"
    return f"{name} = {stat:.2f}, p = {_fmt_p(p)}"

data_dir = os.getenv("DATA_DIR", "./data")
results_dir = os.getenv("RESULTS_DIR", "./results")

# Ensure there is a demographics.csv in data_dir
demographics_path = os.path.join(data_dir, "demographics.csv")
if not os.path.isfile(demographics_path):
    raise FileNotFoundError(f"demographics.csv not found in {data_dir}")

# Load
demographics = pd.read_csv(demographics_path)

# first assert no duplicate subjects
if demographics["Subject"].duplicated().any():
    duplicated_subjects = demographics.loc[demographics["Subject"].duplicated(), "Subject"].unique()
    raise ValueError(f"Duplicated subjects found in demographics.csv: {duplicated_subjects}")

print('-'*80)
print('Demographics:')
print('-'*80)
print(demographics)
print('-'*80)

# Now at each row we transform into:
# <patient int number> <sex> <Clinical diagnosis> <Aetiology> <TPI> \
# <CRS-R ('-'joined subscales)> <LCFS>

# sex as categorical 'M'/'F'
demographics["sex"] = demographics["sex"].map({0: "F", 1: "M"})
# CRS-R subscales we want to join with main column
crsr_sub = ["crsr_aud", "crsr_vis", "crsr_mot", 
            "crsr_mor", "crsr_com", "crsr_vig"]
# Add to CRS-R all the subscales in the format "{CRS-R} ({subscale values joined by '-'})"
def crsr_subs(row):
    if row['Group'] == 'HC':
        return np.nan
    else:
        subscales = row[crsr_sub].astype(int)
        crsr = int(row['CRS-R'])
        return f"{crsr} ({'-'.join(subscales.astype(str))})"
demographics["CRS-R "] = demographics.apply(crsr_subs, axis=1)
demographics.drop(crsr_sub, axis=1, inplace=True)

print('-'*80)
print('Demographics table step 2:')
print('-'*80)
print(demographics)
print('-'*80)



#%% Summaries

def tbi_flagger(x):
    if x is np.nan:
        return np.nan
    elif x == 'Traumatic':
        return 1
    else:
        return 0
demographics["TBI (1 = Traumatic)"] = demographics["aetiology"].map(tbi_flagger)
summary_df = (
    demographics.groupby("Group")
    .agg(
        Count=("Subject", "size"),
        M=("sex", lambda x: (x == 1).sum()),
        F=("sex", lambda x: (x == 0).sum()),
        traumatic=("TBI (1 = Traumatic)", lambda x: (x == 1).sum()),
        nontraum=("TBI (1 = Traumatic)", lambda x: (x == 0).sum()),
        age_median=("age", "median"),
        age_iqr=("age", iqr),
        age_q1=("age", lambda x: x.quantile(0.25)),
        age_q3=("age", lambda x: x.quantile(0.75)),
    )
    .reset_index()
)

def age_fmt(row, use_iqr=True):
    if use_iqr:
        return f"{row['age_median']} ({row['age_iqr']})"
    return f"{row['age_median']} [{row['age_q1']}-{row['age_q3']}]"

def sex_fmt(row):
    return f"{row['M']}/{row['F']}"

def tbi_fmt(row):
    return f"{row['traumatic']}/{row['nontraum']}"

use_iqr_format = True  # Switch to decide format for IQR or Q1-Q3
summary_df["Age, years"] = summary_df.apply(age_fmt, axis=1, use_iqr=use_iqr_format)
summary_df["Sex, M/F"] = summary_df.apply(sex_fmt, axis=1)
summary_df["TBI/non-TBI"] = summary_df.apply(tbi_fmt, axis=1)
# special case: TBI for HC must be np.nan
summary_df.loc[summary_df["Group"] == "HC", "TBI/non-TBI"] = np.nan

summary_df = summary_df[["Group", "Age, years", "Sex, M/F", "TBI/non-TBI"]]

crsr_summary = (
    demographics.query("Group != 'HC'").groupby("Group")
    .agg(
        crsr_median=("CRS-R", "median"),
        crsr_iqr=("CRS-R", iqr),
        crsr_q1=("CRS-R", lambda x: x.quantile(0.25)),
        crsr_q3=("CRS-R", lambda x: x.quantile(0.75)),
        lcfs_median=("LCFS", "median"),
        lcfs_iqr=("LCFS", iqr),
        lcfs_q1=("LCFS", lambda x: x.quantile(0.25)),
        lcfs_q3=("LCFS", lambda x: x.quantile(0.75)),
        event_median=("TPI", "median"),
        event_iqr=("TPI", iqr),
        event_q1=("TPI", lambda x: x.quantile(0.25)),
        event_q3=("TPI", lambda x: x.quantile(0.75)),
    )
    .reset_index()
)

# Add formatted CRS-R and event columns
def format_with_iqr(row, median_col, iqr_col, q1_col, q3_col, use_iqr):
    if use_iqr:
        return f"{row[median_col]} ({row[iqr_col]})"
    return f"{row[median_col]} [{row[q1_col]}-{row[q3_col]}]"

crsr_summary["CRS-R"] = crsr_summary.apply(
    lambda row: format_with_iqr(row, "crsr_median", "crsr_iqr", 
                                "crsr_q1", "crsr_q3", use_iqr_format), 
    axis=1
)

crsr_summary["LCFS"] = crsr_summary.apply(
    lambda row: format_with_iqr(row, "lcfs_median", "lcfs_iqr", 
                                "lcfs_q1", "lcfs_q3", use_iqr_format),
    axis=1
)

crsr_summary["Time from event, days"] = crsr_summary.apply(
    lambda row: format_with_iqr(row, "event_median", "event_iqr", 
                                "event_q1", "event_q3", use_iqr_format), 
    axis=1
)

# Merge CRS-R and event summaries into the summary table
summary_df = summary_df.merge(
    crsr_summary[
        ["Group", "Time from event, days", "CRS-R", "LCFS"]
    ],
    how="left",
    on="Group",
)

# sort for group using the order HC, eMCS, pDoC
group_order = ["HC", "eMCS", "pDoC"]
summary_df["Group"] = pd.Categorical(summary_df["Group"], categories=group_order, ordered=True)
summary_df = summary_df.sort_values("Group")

print("-" * 80)
print("Summary Table:")
print("-" * 80)
print(summary_df.T)
print("-" * 80)

#%%
# =========================
# Statistical testing block
# =========================
apply_fdr = True
df_all = demographics.copy()
df_pat = demographics.copy().query("Group != 'HC'")

# -------------------------
# χ² test: Sex distribution across all groups (HC/eMCS/pDoC)
# -------------------------
sex_ct = pd.crosstab(df_all["Group"], df_all["sex"]).reindex(index=group_order, fill_value=0)
# Ensure both levels present (0=F, 1=M). Add missing columns if necessary.
for col in ['M', 'F']:
    if col not in sex_ct.columns:
        sex_ct[col] = 0
sex_ct = sex_ct[['F', 'M']]  # F, M
chi2_sex, p_sex, _, _ = chi2_contingency(sex_ct.values, correction=False)

# -------------------------
# χ² test: Traumatic vs Non-traumatic (patients only eMCS vs pDoC)
# -------------------------
tbi_flag = "TBI (1 = Traumatic)"
group_order_pat = ["eMCS", "pDoC"]
tbi_ct = pd.crosstab(df_all["Group"], df_all[tbi_flag])
tbi_ct = tbi_ct.reindex(index=group_order_pat, columns=[0.0, 1.0], fill_value=0)
chi2_tbi, p_tbi, _, _ = chi2_contingency(tbi_ct.values, correction=False)

# -------------------------
# Kruskal–Wallis: Age across HC/eMCS/pDoC
# -------------------------
age_groups = [df_all.loc[df_all["Group"] == g, "age"].dropna() for g in group_order]
kw_age = kruskal(*age_groups)
# Pairwise Mann–Whitney for age
mw_age_pairs = {
    "HC vs eMCS": _mw(
        df_all.loc[df_all["Group"] == "HC", "age"], df_all.loc[df_all["Group"] == "eMCS", "age"]
    ),
    "HC vs pDoC": _mw(
        df_all.loc[df_all["Group"] == "HC", "age"], df_all.loc[df_all["Group"] == "pDoC", "age"]
    ),
    "eMCS vs pDoC": _mw(
        df_all.loc[df_all["Group"] == "eMCS", "age"],
        df_all.loc[df_all["Group"] == "pDoC", "age"],
    ),
}

# -------------------------
# eMCS vs pDoC (patients): TPI, CRS-R, LCFS
# -------------------------
mw_TPI = _mw(
    df_pat.loc[df_pat["Group"] == "eMCS", "TPI"], df_pat.loc[df_pat["Group"] == "pDoC", "TPI"]
)
mw_CRSR = _mw(
    df_pat.loc[df_pat["Group"] == "eMCS", "CRS-R"],
    df_pat.loc[df_pat["Group"] == "pDoC", "CRS-R"],
)
mw_LCFS = _mw(
    df_pat.loc[df_pat["Group"] == "eMCS", "LCFS"], df_pat.loc[df_pat["Group"] == "pDoC", "LCFS"]
)

# -------------------------
# Optional FDR (Benjamini–Hochberg) on pairwise tests
# (age pairs + TPI + CRS-R + LCFS). Omnibus tests are reported as-is.
# -------------------------
p_for_fdr = []
keys_for_fdr = []

for k, res in mw_age_pairs.items():
    if res is not None:
        p_for_fdr.append(res.pvalue)
        keys_for_fdr.append(("Age", k))
if mw_TPI is not None:
    p_for_fdr.append(mw_TPI.pvalue)
    keys_for_fdr.append(("TPI", "eMCS vs pDoC"))
if mw_CRSR is not None:
    p_for_fdr.append(mw_CRSR.pvalue)
    keys_for_fdr.append(("CRS-R", "eMCS vs pDoC"))
if mw_LCFS is not None:
    p_for_fdr.append(mw_LCFS.pvalue)
    keys_for_fdr.append(("LCFS", "eMCS vs pDoC"))

fdr_map = {}
if apply_fdr and len(p_for_fdr) > 0:
    _, p_corr, _, _ = multipletests(p_for_fdr, method="fdr_bh")
    for (var, comp), p_adj in zip(keys_for_fdr, p_corr, strict=True):
        fdr_map[(var, comp)] = p_adj

# -------------------------
# Assemble printable / table-ready results
# -------------------------
stats_results = {
    "Sex χ² (HC/eMCS/pDoC)": _fmt_stat("χ²", chi2_sex, p_sex),
    "TBI χ² (eMCS vs pDoC)": _fmt_stat("χ²", chi2_tbi, p_tbi),
    "Age Kruskal-Wallis": _fmt_stat("H", kw_age.statistic, kw_age.pvalue),
}

# Pairwise age (raw + FDR)
for name, res in mw_age_pairs.items():
    if res is None:
        stats_results[f"Age MW {name}"] = "U = NA, p = NA"
    else:
        raw = f"U = {res.statistic:.2f}, p = {_fmt_p(res.pvalue)}"
        adj = f", q = {_fmt_p(fdr_map.get(('Age', name), None))}" if apply_fdr else ""
        stats_results[f"Age MW {name}"] = raw + adj

# eMCS vs pDoC for TPI/CRS-R/LCFS (raw + FDR)
for var, res in [("TPI", mw_TPI), ("CRS-R", mw_CRSR), ("LCFS", mw_LCFS)]:
    if res is None:
        stats_results[f"{var} MW eMCS vs pDoC"] = "U = NA, p = NA"
    else:
        raw = f"U = {res.statistic:.2f}, p = {_fmt_p(res.pvalue)}"
        adj = f", q = {_fmt_p(fdr_map.get((var, 'eMCS vs pDoC'), None))}" if apply_fdr else ""
        stats_results[f"{var} MW eMCS vs pDoC"] = raw + adj

# Pretty print (terminal) and keep a compact dict to drop in your table
print("\n--- Statistical testing ---")
for k, v in stats_results.items():
    print(f"{k}: {v}")

# If you want a compact structure to merge in the final figure/table caption:
stats_for_table = {
    "Sex": f"χ² = {chi2_sex:.3f}, p = {_fmt_p(p_sex)}",
    "Age": f"H = {kw_age.statistic:.2f}, p = {_fmt_p(kw_age.pvalue)}",
    "Age HC-eMCS": stats_results["Age MW HC vs eMCS"],
    "Age HC-pDoC": stats_results["Age MW HC vs pDoC"],
    "Age eMCS-pDoC": stats_results["Age MW eMCS vs pDoC"],
    "TPI eMCS-pDoC": stats_results["TPI MW eMCS vs pDoC"],
    "CRS-R eMCS-pDoC": stats_results["CRS-R MW eMCS vs pDoC"],
    "LCFS eMCS-pDoC": stats_results["LCFS MW eMCS vs pDoC"],
    "TBI eMCS-pDoC": f"χ² = {chi2_tbi:.3f}, p = {_fmt_p(p_tbi)}",
}
print("\n--- Compact stats for table ---")
for k, v in stats_for_table.items():
    print(f"{k}: {v}")

# %%
