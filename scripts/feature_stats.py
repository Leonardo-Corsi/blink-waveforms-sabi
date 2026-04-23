import os
import re
import traceback
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from utils.features import (
    aggregate_features,
    annotate_panels,
    export_significant_html_table,
    libiv,
    logreg_covariate,
    plot_logreg_effects,
    merge_and_check_features,
    plot_features_summary,
    plot_recording_durations,
    run_stat_tests_ncheck,
    run_stat_tests_wilcoxon,
    waveforms_plot,
)

#%% Load opt and environment variables

load_dotenv()  # Load environment variables from a .env file
results_dir = os.getenv('RESULTS_DIR', './results')
data_dir = os.getenv('DATA_DIR', './data')
config_path = os.getenv('CONFIG_PATH', './configs/opt.yml')

# Load options from YAML configuration file
with open(os.path.abspath(config_path)) as file:
    opt = yaml.safe_load(file)
opt = namedtuple('opt', opt.keys())(*opt.values()) # dict2namedtuple

print(f"\nLoaded options from {config_path}:")
_ = [print(f"  - {k}: {v}") for k, v in opt._asdict().items()]


def nonagg_EBR(grouped, start_time, end_time):
    """
    Compute the Eye Blink Rate (EBR) for each group in the grouped DataFrame.
    EBR is defined as the number of blinks per minute.
    """
    all_dfs = []
    for (grp,cnd, sbj), gdf in grouped: 
        
        #print(grp, cnd,sbj,gdf.shape)
        base = f"sub-{sbj}_task-{cnd.upper()}_eog"
        to_load = f"{os.path.dirname(__file__)}/../data/sub-{sbj}/eog/{base}.edf"
        from mne.io import read_raw_edf

        raw = read_raw_edf(to_load,verbose='ERROR')
        remove_movimento_s = 0
        for ann in raw.annotations:
            if ann['description'] == 'Movimento':
                # Find the preceding annotation that is either 'Movimento' or 'OHMETER MEASURE'
                prev_ann = None
                for candidate in reversed([a for a in raw.annotations if a['onset'] < ann['onset']]):
                    if candidate['description'] in ['Movimento', 'OHMETER MEASURE']:
                        prev_ann = candidate
                        break
                
                if prev_ann:
                    # Calculate the delta with the preceding annotation's onset + duration
                    delta_prev_ann = ann['onset'] - (prev_ann['onset'] + prev_ann.get('duration', 0))
                else:
                    # If no preceding annotation, calculate delta with the start
                    delta_prev_ann = ann['onset']
                
                # Add the minimum of 10 seconds or the calculated delta
                remove_movimento_s += min(10, max(0, delta_prev_ann))
            elif ann['description'] == 'OHMETER MEASURE':
                # Add the duration of the OHMETER MEASURE annotation
                remove_movimento_s += ann['duration']
            else: 
                continue
        
        T_min_nocorr = ( (end_time
                          -start_time) 
                        / 60)
        T_min_w_corr = ( (min(end_time, max(raw.times))
                          -max(start_time, min(raw.times))
                          -remove_movimento_s) 
                        / 60 )
        ebr_nocorr = gdf.shape[0] / T_min_nocorr
        ebr_w_corr = gdf.shape[0] / T_min_w_corr
        print(f"[{os.path.basename(to_load)}] - ({min(raw.times):.0f},{max(raw.times):.0f}) out of ({start_time},{end_time}) - {remove_movimento_s}; corr: {ebr_nocorr:.1f} >> {ebr_w_corr}")
        all_dfs.append(pd.DataFrame({'Group':grp,'Condition':cnd,'Subject':sbj,'EBR':ebr_w_corr},index=[0]))
        
    return pd.concat(all_dfs, ignore_index=True)


def nonagg_libiv(grouped, start_time, end_time):
    """
    Compute the Inter-Blink Interval Variability (IBIV) for each group in the grouped DataFrame.
    IBIV is defined as the standard deviation of the logarithm of inter-blink intervals.
    """
    # libiv use bb interval from grouped; we need to remove bb intervals that
    # overlap with artifacts such as:
    # Movimento.onset-10:Movimento.onset 
    # OHMETER.onset:OHMETER.onset+OHMETER.duration
    def _disjoint(a,b,strict=False):
        a, b = sorted(a), sorted(b)
        if strict:
            return a[1] < b[0] or b[1] < a[0]
        else:
            return a[1] <= b[0] or b[1] <= a[0]
        

    all_dfs = []
    for (grp,cnd, sbj), gdf in grouped: 
        libiv_no_corr = libiv(gdf.copy())
        #print(grp, cnd,sbj,gdf.shape)
        base = f"sub-{sbj}_task-{cnd.upper()}_eog"
        to_load = f"{os.path.dirname(__file__)}/../data/sub-{sbj}/eog/{base}.edf"
        from mne.io import read_raw_edf

        raw = read_raw_edf(to_load,verbose='ERROR')
        for ann in raw.annotations:
            if ann['description'] == 'Movimento':
                a = (ann['onset'] - 10, ann['onset'])
                def bb_int(row):
                    if np.isnan(row['BB']):
                        return (row['Time'], row['Time'])
                    else:
                        return (row['Time'] - row['BB'], row['Time'])
                # filter rows of gdf
                gdf = gdf[gdf.apply(lambda row: _disjoint(a, bb_int(row)), 
                                     axis=1)]
            elif ann['description'] == 'OHMETER MEASURE':
                a = (ann['onset'], ann['onset'] + ann['duration'])
                # filter rows of gdf
                gdf = gdf[gdf.apply(lambda row: _disjoint(a, bb_int(row)), 
                                     axis=1)]
            else:
                continue
            
        libiv_w_corr = libiv(gdf)

        print(f"[{os.path.basename(to_load)}] - libiv corr from {libiv_no_corr} to {libiv_w_corr}")
        all_dfs.append(pd.DataFrame({'Group':grp,'Condition':cnd,'Subject':sbj,'LIBIV':libiv_w_corr},index=[0]))

    return pd.concat(all_dfs, ignore_index=True)



#%% Main
if __name__ == '__main__':
    demographics_df = pd.read_csv(os.path.join(data_dir, "demographics.csv"),dtype={"Subject": str})
    subject2group = demographics_df.copy().set_index("Subject")["Group"].to_dict()

    print(f"Working on output folder: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    print("Checking record durations")
    fig = plot_recording_durations(data_dir, subject2group=subject2group,
                                   palette=opt.palette)
    #plt.show(block=False)
    fig.savefig(os.path.join(results_dir,'recording_duration.svg'))
    
    print("Merging and checking features")
    data = merge_and_check_features(os.path.join(results_dir, 
                                                 'eog\\*_features.csv'),
                                    subject2group=subject2group,
                                    filter_similarity=opt.filter_similarity)

    print("Visualizing waveforms")
    fig = waveforms_plot(os.path.join(results_dir, 
                                      'eog\\*_waveforms.csv'),
                         subject2group=subject2group,
                         palette=opt.palette,
                         filter_similarity=opt.filter_similarity
                         )
    #plt.show(block=False)
    fig.savefig(os.path.join(results_dir,'waveforms.svg'))
    
    print('Extracting aggregated features for each subject')
    window_definitions = {"6mins": {'Resting': (30,30+6*60), 'Oddball': (30,30+6*60)}}
    
    features_agg_funcs = {
        "Mean Amplitude (µV)": ('Amplitude', 'mean'),
        "Mean Duration (ms)": ('Duration', 'mean'),
        "Mean Rise Time (ms)": ('Rise', 'mean'),
        "Mean Fall Time (ms)": ('Fall', 'mean'),
    }
    
    features_nonagg_funcs = {
        "EBR (blink/min)": nonagg_EBR,
        "LIBIV": nonagg_libiv,
    }
    
    # first time column 0 for resting, first stimulation for oddball
    records = []
    for sub in data['Subject'].unique():
        for cond in data['Condition'].unique():
            if cond.upper() == 'RESTING':
                first_time = 0
            elif cond.upper() == 'ODDBALL':
                stim_path = os.path.join(results_dir,"STIM", f"sub-{sub}_task-{cond}_eog_stims.csv")
                try:
                    first_time = float(pd.read_csv(stim_path)['onset'].iloc[0])
                except Exception as e:
                    print(f"Failed reading stim file for {sub} at "
                                f"{stim_path}: {e}.")
                    first_time = np.nan
            records.append({'Subject': sub, 'Condition': cond, 'FirstTime': first_time})
    first_time_df = pd.DataFrame.from_records(records)
    data = data.merge(first_time_df, on=['Subject','Condition'], how='left')
    
    dagg = [aggregate_features(data.query(f"Condition == '{cond}'"),
                                    window_name=name,
                                    time_bounds=bounds[cond],
                                    first_time_col='FirstTime',
                                    groupby_columns=['Group','Condition','Subject'],
                                    bound_window_on=['Time'],
                                    aggregate_funcs=features_agg_funcs,
                                    nonaggregate_funcs=features_nonagg_funcs
                                    )
                for name, bounds in window_definitions.items()
                for cond in bounds]

    dagg = pd.concat(dagg, ignore_index=True)

    # Keep only the 6mins window and remove Window column
    dagg = dagg[dagg['Window'] == '6mins'].drop(columns=['Window']).reset_index(drop=True)
    
    print("Imputation of missing data")
    # Imputation of missing values by group, subject, condition, window
    how_many_nans = dagg.isna().sum().sum()
    # Report data to be imputed
    for col in dagg.columns:
        if dagg[col].isna().any():
            nan_rows = dagg[dagg[col].isna()]
            for ridx, _ in nan_rows.iterrows():
                print(f"{pd.DataFrame(dagg.loc[ridx,['Group','Subject','Condition','Window',col]]).T}")
    # Impute with median stratifying by group, condition and window
    for col2impute in [col for col in dagg.columns if col.startswith('std_')]:
        # Impute missing values in the column col2impute
        dagg[col2impute] = dagg.groupby(['Group', 'Condition', 'Window'])[col2impute].transform(
            lambda x: x.fillna(x.median()))
    print(f"Imputed {how_many_nans} NaN values, remaining NaNs: {dagg.isna().sum().sum()}")

    dagg['Group'] = pd.Categorical(dagg['Group'], categories=opt.palette.keys(), ordered=True)
    dagg = dagg.sort_values(by=['Group', 'Subject', 'Condition'], 
                                ascending=[True, True, False])
    
    # sort columns 
    first_cols = ['Group', 'Subject', 'Condition']
    dagg = dagg[ first_cols +
                    # with EBR and LIBIV first if present 
                    [col for col in dagg.columns
                        if col not in first_cols and
                        col.startswith(('EBR (blink/min)', 'LIBIV'))] +
                    # then the rest
                    [col for col in dagg.columns
                        if col not in first_cols and
                        not col.startswith(('EBR (blink/min)', 'LIBIV'))]
                    ]

    dagg.to_csv(os.path.join(results_dir, 'aggregated_features.csv'),
                index=False)
    print(f'Aggregated features, saved in {os.path.join(results_dir, "aggregated_features.csv")}:')
    print(dagg)
    
    
    # Join aggregated results with demographics
    dagg_and_demo = dagg.merge(demographics_df, on=['Group', 'Subject'], how='left')
    if 'Window' in dagg_and_demo.columns:
        dagg_and_demo = dagg_and_demo.drop(columns=['Window'])
    print(f"Results with demographics: {dagg_and_demo.shape[0]} rows, {dagg_and_demo.shape[1]} columns")

    # Save joined results
    out_path = os.path.join(results_dir, 'aggregated_features_with_demographics.csv')
    dagg_and_demo.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Fold conditions into columns
    demo_cols = [c for c in demographics_df.columns if c in dagg_and_demo.columns]
    value_cols = [c for c in dagg_and_demo.columns if c not in set(['Group', 'Subject', 'Condition']) | set(demo_cols)]
    dagg_and_demo_folded = (
        dagg_and_demo.copy().fillna(-1, inplace = False) # needed to pivot
        .pivot_table(index=demo_cols, columns='Condition', values=value_cols, aggfunc='first')
        .reset_index()
    )
    dagg_and_demo_folded.replace(-1, np.nan, inplace=True)

    # Flatten MultiIndex columns like "<Condition> <Feature>"
    new_cols = []
    for col in dagg_and_demo_folded.columns:
        if isinstance(col, tuple):
            feat, cond = col
            new_cols.append(f"{cond} {feat}".strip())
        else:
            new_cols.append(col)
    dagg_and_demo_folded.columns = new_cols

    # Revert placeholder demographics back to NaN
    dagg_and_demo_folded[demo_cols] = dagg_and_demo_folded[demo_cols].replace(-1, pd.NA)

    print(f"Results with demographics and folded conditions: {dagg_and_demo_folded.shape[0]} rows, {dagg_and_demo_folded.shape[1]} columns")

    # Save folded results
    out_path_folded = os.path.join(results_dir, 'aggregated_features_folded_with_demographics.csv')
    dagg_and_demo_folded.to_csv(out_path_folded, index=False)
    print(f"Saved: {out_path_folded}")
    
    
    group_stats_df = run_stat_tests_ncheck(dagg, 
                            condition_col='Condition', 
                            group_col='Group', 
                            alpha=0.05, 
                            p_adjust='holm',
                            group_comparisons=[('HC', 'eMCS'), 
                                               ('eMCS', 'pDoC'), 
                                               ('HC', 'pDoC')])
    
    print('\nStatistical test results:\n')
    group_stats_df = group_stats_df.loc[:, [col for col in group_stats_df.columns 
                                            if col[1] != 'Statistic']]
    group_stats_df = group_stats_df.applymap(lambda x: round(x, 4) 
                                             if isinstance(x, float) else x)
    
    cols_mdn_iqr_nrm = [c for c in group_stats_df.columns if c[0].startswith('Median')]
    group_stats_df_mdn_iqr_nrm = group_stats_df[cols_mdn_iqr_nrm]
    print("Median, IQR, and Normality test results:\n")
    print(group_stats_df_mdn_iqr_nrm.to_csv(sep=';'))

    cols_kruskal_ph = [c for c in group_stats_df.columns if not c[0].startswith('Median') and not c[0].startswith('Normality')]
    group_stats_df_kruskal_ph = group_stats_df[cols_kruskal_ph]
    print("Kruskal-Wallis and Conover post-hoc test results:\n")
    group_stats_df_kruskal_ph = group_stats_df_kruskal_ph.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)
    print(group_stats_df_kruskal_ph.to_csv(sep=';'))

    export_significant_html_table(group_stats_df,
                                os.path.join(results_dir, 'KW-Dunn_test.html'),
                                hlines=[4,6,8,10,12],
                                vlines=[])
    
    try:
        
        wilcoxon_stats_df = run_stat_tests_wilcoxon(dagg, 
                                                    split_by='Group', 
                                                    pair_by='Condition')
        print("\nWilcoxon Signed-Rank Test Results:\n")

        wilcoxon_stats_df.index.name = None
        wilcoxon_stats_df = wilcoxon_stats_df.applymap(lambda x: round(x, 2) 
                                                       if isinstance(x, float) else x)
        
        print(wilcoxon_stats_df.to_csv(sep=';'))
        
    except Exception as e:
        print(f"Error occurred while processing Wilcoxon test: {e}")
        print(traceback.format_exc())
        
    plot_layout = np.array([
            [('6mins', 'EBR (blink/min)'),  ('6mins', 'LIBIV'), ('6mins', 'Mean Amplitude (µV)'),
            ],
            [ ('6mins', 'Mean Duration (ms)'), ('6mins', 'Mean Rise Time (ms)'), ('6mins', 'Mean Fall Time (ms)')
            ],
        ])

    # Plot the features summary for each group and condition
    plot_features_summary(dagg.query("Condition == 'Resting' or Condition == 'Oddball'"), 
                          plot_layout,
                          palette=opt.palette)

    # hack for duration fall time and rise time axes
    plt.gcf().get_axes()[-1].sharey(plt.gcf().get_axes()[-2]) # rise and fall time share y with duration
    plt.gcf().set_size_inches(7.7, 5.5)
    [ax.set_title('') for ax in plt.gcf().get_axes()]

    # Add statistical annotations to the boxplots
    from statannotations.Annotator import Annotator
    # Define pairs for comparisons
    pairs = [
        (("HC", "Resting"), ("eMCS", "Resting")),
        (("HC", "Resting"), ("pDoC", "Resting")),
        (("eMCS", "Resting"), ("pDoC", "Resting")),
        (("HC", "Oddball"), ("eMCS", "Oddball")),
        (("HC", "Oddball"), ("pDoC", "Oddball")),
        (("eMCS", "Oddball"), ("pDoC", "Oddball")),
    ]

    # Add annotations to the features summary plot
    fig = plt.gcf()
    axes = fig.get_axes()
    for ax, (window, feature) in zip(axes, plot_layout.reshape(-1, 2), strict=True):
        if window is None or feature is None:
            continue
        # Filter data for the current feature and window
        data_to_annotate = dagg[['Group', 'Condition', feature]]
        data_to_annotate = data_to_annotate.rename(columns={feature: "Value"})

        # Initialize Annotator
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=data_to_annotate,
            x="Group",
            hue="Condition",
            y="Value",
            order=["HC", "eMCS", "pDoC"],
            hue_order=["Resting", "Oddball"],
            perform_stat_test=False,  # We will set p-values manually
            verbose=False,
        )
        
        # Get post-hoc test results
        custom_pvalues = []
        for pair in pairs:
            group1, condition1 = pair[0]
            group2, condition2 = pair[1]
            feature_name = feature.split('[')[0].strip()

            # Extract the p-value
            try:
                p_value = group_stats_df.loc[(feature_name, condition1), (f"Conover", f"{group1} vs. {group2}")]
                if isinstance(p_value, str):
                    match = re.findall(r'(0\.\d+)', p_value)
                    p_value = float(match[1]) if match else np.nan# take the adjusted one
                    
                else:
                    p_value = np.nan  # If the p-value is not found, set it to NaN
            except KeyError:
                p_value = np.nan  # If the p-value is not found, set it to NaN

            if pd.isna(p_value) or p_value == '':
                p_value = 1
            # Add p-value to the list
            custom_pvalues.append(p_value)

        # Ensure the number of custom_pvalues matches the number of pairs
        if len(custom_pvalues) != len(pairs):
            raise ValueError("Mismatch between the number of pairs and custom p-values.")


        # Customize annotation format
        # Ensure the annotator object is properly initialized
        if not hasattr(annotator, "configure"):
            raise AttributeError("The annotator object is not properly initialized. Ensure it is created with a valid plotter.")

        # Configure the annotator with the correct parameters
        annotator.configure(
            text_format="star",
            loc="inside",
            hide_non_significant=True,
        )
        print(f"Adding annotations: { {pair: p 
              for pair, p in zip(pairs, custom_pvalues, strict=True) } }")

        # Apply the p-values and annotate
        if custom_pvalues:
            annotator.set_pvalues_and_annotate(custom_pvalues)
        else:
            raise ValueError("Custom p-values are missing. Ensure they are calculated and passed correctly.")

    annotate_panels(plt.gcf().get_axes(),xy =(-0.35, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(results_dir,'features_boxplot.svg')))
    plt.show(block=False)
    #'''
    
    
    # Logistic regression with age/sex covariates
    try:
        # Work on merged features + demographics
        df_lr = dagg_and_demo.copy()

        # Find and standardize age/sex columns
        def _find_col(df, names):
            for n in names:
                if n in df.columns:
                    return n
            for c in df.columns:
                if c.lower() in {n.lower() for n in names}:
                    return c
            return None

        age_src = _find_col(df_lr, ['age', 'Age', 'AGE'])
        sex_src = _find_col(df_lr, ['sex', 'Sex', 'SEX', 'Gender', 'gender'])

        if age_src is None or sex_src is None:
            print("Age/sex columns not found in demographics. Skipping logistic regression.")
        else:
            # Normalize age to numeric
            df_lr['age'] = pd.to_numeric(df_lr[age_src], errors='coerce')

            # Normalize sex to numeric (0/1), try common string encodings first
            if df_lr[sex_src].dtype == object:
                sex_map = {
                    'm': 1, 'male': 1, 'man': 1, 'maschio': 1,
                    'f': 0, 'female': 0, 'woman': 0, 'femmina': 0
                }
                sex_norm = (
                    df_lr[sex_src].astype(str).str.strip().str.lower().map(sex_map)
                )
                # Fallback: attempt numeric coercion
                sex_norm = sex_norm.where(~sex_norm.isna(),
                                          pd.to_numeric(df_lr[sex_src], errors='coerce'))
                df_lr['sex'] = sex_norm
            else:
                df_lr['sex'] = pd.to_numeric(df_lr[sex_src], errors='coerce')

            # Ensure required columns
            features2screen = [
                'EBR (blink/min)',
                'LIBIV',
                'Mean Amplitude (µV)',
                'Mean Duration (ms)',
                'Mean Rise Time (ms)',
                'Mean Fall Time (ms)'
            ]
            features2screen = [f for f in features2screen if f in df_lr.columns]
            if not features2screen:
                print("No matching features found for logistic regression. Skipping.")
            else:
                # Run per condition
                def _run_condition(cond):
                    sub = df_lr.query("Condition == @cond").copy()
                    # Drop rows missing essential covariates or target
                    sub = sub.dropna(subset=['age', 'sex', 'Group'])
                    if sub.empty:
                        return None
                    dfs = []
                    for feat in features2screen:
                        sub_feat = sub.dropna(subset=[feat])
                        if sub_feat.empty:
                            continue
                        try:
                            res = logreg_covariate(sub_feat, feat, ['age', 'sex'], 'Group', 
                                                   verbose=True, plot=False)
                            figro = plot_logreg_effects(res, feat, ['age','sex'],'Group', sub, 
                                                      group2col = {g:d['Resting'] 
                                                                   for g,d in opt.palette.items()},
                                                      show_prob=False
                                                      )
                            figmp = plot_logreg_effects(res, feat, ['age','sex'],'Group', sub, 
                                                      group2col = {g:d['Resting'] 
                                                                   for g,d in opt.palette.items()},
                                                      show_prob=True
                                                      )
                            os.makedirs(results_dir+'/logreg_plots/', exist_ok=True)
                            # save relative odds fig
                            featpath = feat.replace(" (blink/min)", "").replace(" ","_")
                            figro.savefig(os.path.join(results_dir+'/logreg_plots/',
                                                       f'logreg_{cond}_{featpath}_odds_ratio.svg'))
                            # save mapped probabilities fig
                            figmp.savefig(os.path.join(results_dir+'/logreg_plots/',
                                                       f'logreg_{cond}_{featpath}_mapped_probabilities.svg'))
                            #plt.show(block=True)
                            # Attach feature name to index if needed
                            if isinstance(res, pd.Series):
                                res = res.to_frame().T
                                res.index = pd.Index([feat], name='Feature')
                            dfs.append(res)
                        except Exception as _e:
                            print(f"logreg failed for {cond} / {feat}: {_e}")
                    if not dfs:
                        return None
                    out = pd.concat(dfs, axis=0, ignore_index=False)
                    out['Condition'] = cond
                    return out

                lr_age_df_resting = _run_condition('Resting')
                lr_age_df_oddball = _run_condition('Oddball')

                frames = [d for d in [lr_age_df_resting, lr_age_df_oddball] if d is not None]
                if not frames:
                    print("No logistic regression output produced.")
                else:
                    lr_age_df = pd.concat(frames, axis=0, ignore_index=False)

                    # Formatting table
                    req_cols = {
                        'feat_OR', 'feat_CI_lower', 'feat_CI_upper', 
                        'feat_p-value', 'feat_R2_Nagelkerke', 'feat_AUC',
                        'feat_corrected_OR', 'feat_corrected_CI_lower', 'feat_corrected_CI_upper',
                        'feat_corrected_p-value', 'feat_corrected_R2_Nagelkerke', 'feat_corrected_AUC'
                    }
                    if req_cols.issubset(set(lr_age_df.columns)):
                        lr_age_df_fmt = lr_age_df.round(4).assign(
                            Uncorrected=lr_age_df['feat_OR'].round(4).astype(str) + " (" +
                                        lr_age_df['feat_CI_lower'].round(4).astype(str) + ", " +
                                        lr_age_df['feat_CI_upper'].round(4).astype(str) + ")",
                            Corrected=lr_age_df['feat_corrected_OR'].round(4).astype(str) + " (" +
                                       lr_age_df['feat_corrected_CI_lower'].round(4).astype(str) + ", " +
                                       lr_age_df['feat_corrected_CI_upper'].round(4).astype(str) + ")"
                        )
                        lr_age_df_fmt = lr_age_df_fmt[['Condition', 
                                                       'Uncorrected', 'feat_p-value', 
                                                       'feat_R2_Nagelkerke', 'feat_AUC',
                                                       'Corrected', 'feat_corrected_p-value', 
                                                       'feat_corrected_R2_Nagelkerke', 'feat_corrected_AUC']]

                        # Prepare MultiIndex table
                        lr_age_df_midx = lr_age_df_fmt.copy().reset_index(names=['Feature'])
                        featorder = [feat.split(' [')[0] for feat in features2screen]
                        # Enforce sorting by Feature (using features2screen order) and then by Condition
                        lr_age_df_midx['Feature'] = pd.Categorical(lr_age_df_midx['Feature'],
                                                                  categories=featorder,
                                                                  ordered=True)
                        lr_age_df_midx.sort_values(by=['Feature', 'Condition'], 
                                                  ascending=[True, False],
                                                  inplace=True)
                        lr_age_df_midx = lr_age_df_midx.set_index(
                            pd.MultiIndex.from_arrays(
                                [lr_age_df_midx['Feature'].values, 
                                 lr_age_df_midx['Condition'].values],
                                names=['Feature', 'Condition']
                            )
                        ).drop(columns=['Feature', 'Condition'])

                        lr_age_df_midx.columns = pd.MultiIndex.from_tuples([
                            ('Uncorrected', 'OR, median (CI 95%)'),
                            ('Uncorrected', 'p'),
                            ('Uncorrected', 'Nagelkerke R2'),
                            ('Uncorrected', 'AUC'),
                            ('Corrected', 'OR, median (CI 95%)'),
                            ('Corrected', 'p'),
                            ('Corrected', 'Nagelkerke R2'),
                            ('Corrected', 'AUC')
                        ])

                        print("Logistic regression results:")
                        print(lr_age_df_midx.to_csv(sep=';'))

                        # Save
                        out_csv = os.path.join(results_dir, 'logreg_covariate_results.csv')
                        lr_age_df_midx.to_csv(out_csv)
                        print(f"Saved: {out_csv}")
                    else:
                        # Fallback: save raw output
                        print("Unexpected columns from logreg_covariate; saving raw output.")
                        out_csv = os.path.join(results_dir, 'logreg_covariate_results_raw.csv')
                        lr_age_df.to_csv(out_csv)
                        print(f"Saved: {out_csv}")

    except Exception as e:
        print(f"Error during logistic regression analysis: {e}")
        print(traceback.format_exc())




    
    
