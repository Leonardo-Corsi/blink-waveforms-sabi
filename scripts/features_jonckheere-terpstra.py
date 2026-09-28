import os
from itertools import combinations

import numpy as np
import pandas as pd
import scikit_posthocs as skph
from dotenv import load_dotenv
from jonckheere_test import jonckheere_test
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GROUP_ORDER = ["HC", "eMCS", "MCS", "UWS"]
ALTERNATIVE = "two-sided"


def subject_key(series):
    n = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    return n.where(n.isna(), n.astype("Int64").astype(str))


def fmt_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return f"{p:.3f}***"
    if p < 0.01:
        return f"{p:.3f}**"
    if p < 0.05:
        return f"{p:.3f}*"
    return f"{p:.3f}"


def median_iqr(values, normality_p):
    values = values.dropna()
    if values.empty:
        return ""
    q1, q3 = np.percentile(values, [25, 75])
    marker = "*" if normality_p is not None and normality_p < 0.05 else ""
    return f"{np.median(values):.1f} ({q1:.1f}, {q3:.1f}){marker}"


def stat_table(data, numeric_features, comparisons):
    rows = []
    for feature in numeric_features:
        label = feature.split("[")[0].strip()
        for condition, cond_data in data.groupby("Condition", sort=False):
            row = {("Feature", "Condition"): (label, condition)}
            groups = []
            for group in GROUP_ORDER:
                values = cond_data.loc[cond_data["JT_Group"] == group, feature].dropna()
                p_norm = stats.shapiro(values).pvalue if len(values) >= 3 else None
                row[("Median (IQR)", group)] = median_iqr(values, p_norm)
                row[("Normality Test", f"{group} p")] = p_norm
                if len(values) > 0:
                    groups.append(values)

            if len(groups) >= 2:
                h_stat, p_kw = stats.kruskal(*groups)
                row[("Kruskal-Wallis", f"H (df={len(groups) - 1})")] = h_stat
                row[("Kruskal-Wallis", "p")] = fmt_p(p_kw)
                row[("Kruskal-Wallis", "eta2")] = h_stat / (len(cond_data) - 1)
                if p_kw < 0.05:
                    posthoc_data = cond_data[["JT_Group", feature]].dropna()
                    ph = skph.posthoc_conover(posthoc_data, val_col=feature, group_col="JT_Group")
                    ph_adj = skph.posthoc_conover(
                        posthoc_data, val_col=feature, group_col="JT_Group", p_adjust="holm"
                    )
                    for g1, g2 in comparisons:
                        if g1 in ph.index and g2 in ph.columns:
                            row[("Conover", f"{g1} vs. {g2}")] = (
                                f"{fmt_p(ph.loc[g1, g2])} ({fmt_p(ph_adj.loc[g1, g2])})"
                            )

            for g1, g2 in comparisons:
                row.setdefault(("Conover", f"{g1} vs. {g2}"), "")
            rows.append(row)

    out = pd.DataFrame(rows).set_index(("Feature", "Condition"))
    out.index = pd.MultiIndex.from_tuples(out.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def add_ordered_group(features, demographics):
    demo = demographics[["Subject", "Clinical diagnosis"]].copy()
    features = features.copy()
    features["_SubjectKey"] = subject_key(features["Subject"])
    demo["_SubjectKey"] = subject_key(demo["Subject"])
    out = features.merge(demo[["_SubjectKey", "Clinical diagnosis"]], on="_SubjectKey", how="left")
    diagnosis = out["Clinical diagnosis"].where(out["Clinical diagnosis"].isin(GROUP_ORDER))
    out["JT_Group"] = diagnosis.fillna(out["Group"])
    out = out[out["JT_Group"].isin(GROUP_ORDER)].copy()
    out["_GroupOrder"] = pd.Categorical(out["JT_Group"], categories=GROUP_ORDER, ordered=True)
    out = out.sort_values(["_GroupOrder", "Subject", "Condition"]).drop(columns=["_GroupOrder"])
    return out


def add_jt_columns(stats_df, data, numeric_features):
    rank = {group: i + 1 for i, group in enumerate(GROUP_ORDER)}
    stats_df[("Jonckheere-Terpstra", "J")] = np.nan
    stats_df[("Jonckheere-Terpstra", "p")] = ""

    for feature in numeric_features:
        label = feature.split("[")[0].strip()
        for condition, cond_data in data.groupby("Condition", sort=False):
            test_data = cond_data[["JT_Group", feature]].dropna()
            if test_data["JT_Group"].nunique() < 2:
                continue
            result = jonckheere_test(
                test_data[feature].to_numpy(float),
                test_data["JT_Group"].map(rank).to_numpy(int),
                alternative=ALTERNATIVE,
            )
            stats_df.loc[(label, condition), ("Jonckheere-Terpstra", "J")] = result.statistic
            stats_df.loc[(label, condition), ("Jonckheere-Terpstra", "p")] = fmt_p(result.p_value)

    return stats_df


if __name__ == "__main__":
    load_dotenv()
    results_dir = os.getenv("RESULTS_DIR", os.path.join(ROOT, "results"))
    data_dir = os.getenv("DATA_DIR", os.path.join(ROOT, "data"))

    features = pd.read_csv(os.path.join(results_dir, "aggregated_features.csv"), dtype={"Subject": str})
    demographics = pd.read_csv(os.path.join(data_dir, "demographics.csv"), dtype={"Subject": str})
    data = add_ordered_group(features, demographics)

    numeric_features = [
        col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])
    ]
    present_groups = [g for g in GROUP_ORDER if g in set(data["JT_Group"])]
    comparisons = list(combinations(present_groups, 2))

    stats_df = stat_table(data, numeric_features, comparisons)
    stats_df = add_jt_columns(stats_df, data, numeric_features)

    conover_cols = [col for col in stats_df.columns if col[0] == "Conover"]
    stats_df = stats_df[[col for col in stats_df.columns if col[0] != "Conover"] + conover_cols]
    stats_df = stats_df.map(lambda x: round(x, 4) if isinstance(x, float) else x)

    out_path = os.path.join(results_dir, "PR01_jonckheere_terpstra.csv")
    stats_df.to_csv(out_path)
    print(f"Ordered groups: {GROUP_ORDER}; alternative={ALTERNATIVE}")
    print(pd.crosstab(data["JT_Group"], data["Condition"]).reindex(GROUP_ORDER))
    print(f"Saved: {out_path}")
