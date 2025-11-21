import itertools
import os
import warnings
from glob import glob
from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import scikit_posthocs as skph
import scipy
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator
from mne import set_log_level
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

set_log_level('ERROR') # Set MNE log level to ERROR to avoid excessive output
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.spines.right': False,
    'axes.spines.top': False
})
plt.switch_backend('TkAgg') 

def plot_recording_durations(folder, subject2group=None,palette=None):
    recording_duration = get_recording_duration(edf_folder=folder) 
    if subject2group is not None: # force group assignment
        recording_duration['Group'] = recording_duration['Subject'].map(subject2group)

    # quick plot of recording duration in minutes
    fig,ax = plt.subplots(1,1,figsize=(7, 5),
                        num='Recording time')
    recording_duration_m = recording_duration.copy()
    recording_duration_m['max_time'] = (recording_duration_m['max_time']/60)
    
    # plot resting
    palette_resting = {key: palette[key]['Resting'] for key in palette}
    sns.swarmplot(recording_duration_m.query("Condition == 'Resting'"),
                palette=palette_resting,
                x='Condition',
                hue='Group',
                y='max_time',
                size=5,
                marker='d',
                ax=ax)
    # plot oddball
    palette_oddball = {key: palette[key]['Oddball'] for key in palette}
    sns.swarmplot(recording_duration_m.query("Condition == 'Oddball'"),
                palette=palette_oddball,
                x='Condition',
                hue='Group',
                y='max_time',
                size=5,
                marker='d',
                ax=ax)
    # decorate
    ax.set_ylim(0,ax.get_ylim()[1])
    ax.set_ylabel('Recording time (min)')
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))
    ax.grid(axis='y',which='major', linestyle='-', linewidth=0.75, color='gray')
    ax.grid(axis='y',which='minor', linestyle='--', linewidth=0.5, color='lightgray')
    handles, labels = ax.get_legend_handles_labels()
    labels[:3] = [lbl+', Resting' for lbl in labels[:3]]
    labels[3:] = [lbl+', Oddball' for lbl in labels[3:]]
    ax.legend(handles, labels, loc='lower center', ncol=2,)
    plt.tight_layout()
    
    return fig

def similarity_knee_filter(target_df, similarity_df):
    
    if target_df.empty or similarity_df.empty:
        warnings.warn("Empty DataFrame passed to similarity_knee_filter, returning empty DataFrame.", stacklevel=2)
        return target_df
    
    if target_df.shape[0] != similarity_df.shape[0]:
        raise ValueError(f"df.shape must be the same, instead is {target_df.shape[0]} and {similarity_df.shape[0]}.")
    
    # Construct the CDF of similarity scores
    cdf = similarity_df['Similarity'].value_counts(normalize=True).sort_index().cumsum()
    sim = np.hstack([0,cdf.index])
    cts = np.hstack([0,cdf.values]) # cumulative counts
    itp = scipy.interpolate.interp1d(sim,cts,kind='linear',fill_value="extrapolate")
    s = np.arange(0,max(sim),1e-3) # max sim
    cdf_smooth = scipy.signal.savgol_filter(itp(s),50,2).reshape(-1,1) # 50e-3 i.e. 5% similarity
    dynprog_model = rpt.Dynp(model='l1').fit(cdf_smooth)
    bkps = dynprog_model.predict(n_bkps=3) # first breakpoint

    if len(bkps) > 1: # otherwise means they were not found
        # Apply the knee point filter
        return target_df.loc[similarity_df['Similarity'] >= s[bkps[0]],:]
    else:
        print("No breakpoint found, returning original DataFrame.")
        return target_df

def merge_and_check_features(file_pattern,subject2group=None, filter_similarity=False):

    feature_files = glob(file_pattern)
    data_loaded = []
    for ff in tqdm(feature_files,desc="Loading feature files"):
        df = pd.read_csv(ff,dtype={'Subject':str})
        if filter_similarity:
            sim_df = pd.read_csv('_'.join(ff.split('_')[:-1]) + '_similarity.csv')
            data_loaded.append(similarity_knee_filter(df, sim_df))
        else:
            data_loaded.append(df)
            
    data_loaded = pd.concat(data_loaded,
                            ignore_index=True)
    # Group
    if subject2group is not None:
        data_loaded['Group'] = data_loaded['Subject'].map(subject2group)
    data_loaded['Condition'] = data_loaded['Condition'].str.capitalize()
    
    # Conventional measurement unit
    data_loaded['Amplitude'] *= 1e6 # to uV
    data_loaded['Duration'] *= 1e3 # to ms
    data_loaded['Rise'] *= 1e3 # to ms
    data_loaded['Fall'] *= 1e3 # to ms
    
   

    # Check that each subject is associated with a single group
    subjects_groups = data_loaded.groupby('Subject')['Group'].nunique()
    if (subjects_groups > 1).any():
        print("Subjects with multiple group assignments:")
        print(subjects_groups[subjects_groups > 1])
    else:
        print("All subjects have a single group assignment.")
        
    # Check number of subjects per group
    subjects_per_group = data_loaded.groupby('Group')['Subject'].nunique()
    print("\nNumber of subjects per group:")
    print(subjects_per_group)
    
    # For each group, check if each subject has both Resting and Oddball conditions
    print("\nChecking if each subject has both Resting and Oddball conditions")
    for grp, grp_df in data_loaded.groupby('Group'):
        all_ok = True
        subj_cond = grp_df.groupby('Subject')['Condition'].apply(set)
        for subj, conds in subj_cond.items():
            conds = {str(c).capitalize() for c in conds if pd.notnull(c)}
            missing = {'Resting', 'Oddball'} - conds
            if missing:
                print(f"Subject {subj} in group {grp} is missing: {missing}")
                all_ok = False
            else:
                continue#print(f"Subject {subj} in group {grp} has both conditions.")
        if all_ok:
            print(f"All subjects in group {grp} have both Resting and Oddball conditions.")
            
    return data_loaded

def _load_waveform_df(csv_path):
    wf_df = pd.read_csv(csv_path,index_col=0)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    sub, condition = base.split('_')[:2]
    if 'sub-' in sub:
        sub = sub.replace('sub-','')
    else:
        raise ValueError(f" 'sub-' not found in filename {base}")
    if 'task-' in condition:
        condition = condition.replace('task-','')
    else:
        raise ValueError(f" 'task-' not found in filename {base}")
    
    subject = sub
    condition = condition.capitalize()
    # bring index (peak location in s, 'Peak_s' to a column)
    wf_df.reset_index(inplace=True)
    # Go to long-form with 'Time' and 'Amplitude' columns, plus the Peak_s column 
    wf_df = wf_df.melt(id_vars=['Peak_s'], 
                       var_name='Time', value_name='Amplitude')
    wf_df['Time'] = wf_df['Time'].astype(float)
    # Add Subject, Condition columns at the beginning
    wf_df.insert(0, 'Subject', subject)
    wf_df.insert(1, 'Condition', condition)
    # Sort by Group (inverse), subject, condition, Peak_s, Time
    wf_df.sort_values(by=['Subject', 'Condition', 'Peak_s', 'Time'], 
                      ascending=[True, True, True, True], inplace=True)
    
    return wf_df

def waveforms_plot(file_pattern, subject2group=None, palette=None, filter_similarity=False):
    
    waveforms_files = glob(file_pattern)
    waveforms_merged = []
    for wf in tqdm(waveforms_files, desc="Loading waveform files"):
        df = _load_waveform_df(wf)
        if filter_similarity:
            sim_df = pd.read_csv('_'.join(wf.split('_')[:-1]) + '_similarity.csv')
            sim_df['Peak_s'] = df.query("Time == 0")['Peak_s'].values # same length
            sim_df = df['Peak_s'].to_frame().merge(sim_df,on='Peak_s',how='left') # broadcast
            waveforms_merged.append(similarity_knee_filter(df, sim_df))
        else:
            waveforms_merged.append(df)

    waveforms_merged = pd.concat(waveforms_merged, ignore_index=True)
    
    waveforms_merged['Group'] = waveforms_merged['Subject'].map(subject2group)
    
    # Make categorical following the order of custom_palette
    group_cats = palette.keys() if palette else waveforms_merged['Group'].unique()
    waveforms_merged['Group'] = pd.Categorical(waveforms_merged['Group'],
                                                categories=group_cats,
                                                ordered=True)
    waveforms_merged.sort_values(by=['Group', 'Condition', 'Subject', 'Peak_s','Time'], 
                                ascending=[True, False, True, True, True],
                                inplace=True)
    
    waveforms_merged['Time'] *= 1e3 # to ms
    waveforms_merged['Amplitude'] *= 1e6 # to uV
    
    np.random.seed(42)  # For reproducibility
    exemplary_idx = np.random.randint(0, 8)  # Randomly select an exemplary subject index
    print(f"Exemplary subject index for single-trial plots: {exemplary_idx}")
    conditions = sorted(
            waveforms_merged['Condition'].dropna().unique(),
            reverse=True
        )
    fig, axs = plt.subplots(2,6, figsize=(7.7, 4.5),sharex=True,sharey=True,)
    for j, grp in enumerate(group_cats):
        for i, cond in enumerate(conditions):
            
            # First row for averaged waveforms
            ax = axs[0, i + j*2]  
            # Filter the data for the specific group and condition
            df_plot = waveforms_merged.query("Group == @grp and Condition == @cond")
            # Plot each subject's waveform
            for subject in tqdm(df_plot['Subject'].unique(), desc=f"Plotting for group {grp} and condition {cond}"):
                sub_wf = df_plot.query("Subject == @subject").copy()
                # Fill NaNs only on float columns
                float_cols = sub_wf.select_dtypes(include=['float']).columns
                if len(float_cols):
                    sub_wf[float_cols] = sub_wf[float_cols].fillna(0)
                
                sns.lineplot(data=sub_wf,
                            x='Time', y='Amplitude',
                            #errorbar=('ci', 50), 
                            estimator='median',
                            #err_style='band',
                            linewidth=0.75, ax=ax, alpha=0.75,
                            label=subject,
                            color=get_palette(palette, grp, cond, 'gray'),
                            legend=False)
            #ax.set_title(f"{grp}\n{cond}",fontsize=6)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_xlim(-360, 360)  
            ax.set_ylim(-120, 920)  
            
            # Annotate the panel letter at top left corner
            ax.annotate(chr(ord('A') + j*2 + i), xy=(-0.25, 1.05), xycoords='axes fraction',
                        fontsize=14, weight='bold',color='black', ha='left', va='top')
            
            # Second row for an exemplary subject
            ax = axs[1, i + j*2]
            # Filter the data for the specific group, condition, and subject
            exemplary_subject = df_plot['Subject'].unique()[exemplary_idx]  # Take the first subject as an example
            sub_wf = df_plot.query("Subject == @exemplary_subject")
            sns.lineplot(data=sub_wf,
                        x='Time', y='Amplitude',
                        estimator=None, units='Peak_s',
                        #errorbar='se', err_style='band',
                        linewidth=0.25, ax=ax, alpha=0.5,
                        label=exemplary_subject,
                        color=get_palette(palette, grp, cond, 'gray'),
                        legend=False)
            #ax.set_title(f"{grp}\n{cond} - {exemplary_subject}", fontsize=6)
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Amplitude [µV]")
            ax.set_xlim(-360, 360)  
            ax.set_ylim(-120, 920)  
            
            # Annotate the panel letter at top left corner
            ax.annotate(chr(ord('A') + j*2 + i + 6), xy=(-0.25, 1.05), xycoords='axes fraction',
                        fontsize=14, weight='bold',color='black', ha='left', va='top')
    plt.tight_layout()
    return fig

def aggregate_features(data,              
                        groupby_columns = ['Group','Condition','Subject'],
                        bound_window_on = ['Time'],
                        window_name='Full',
                        time_bounds=(-np.inf, np.inf),
                        first_time_col=None,
                        aggregate_funcs = {},
                        nonaggregate_funcs = {}
                        ):

    # Validation: aggregate_funcs must be a dict with tuples of (column_name, agg_func)
    if (aggregate_funcs is None
        or (isinstance(aggregate_funcs, dict) and len(aggregate_funcs) == 0)
       ):
        # Defaults to aggregation by mean for all numeric columns
        aggregable_cols = [c for c in data.columns
                           if c not in groupby_columns + bound_window_on
                           and data[c].isnumeric().all()]
        aggregate_funcs = {f"mean_{col}": (col,'mean') for col in aggregable_cols}
    elif isinstance(aggregate_funcs, dict):
        # Check if all values in aggregate_funcs are tuples of (column_name, agg_func)
        # and that 
        # 1) column_name is a valid column in data
        # 2) agg_func is str in ['mean', 'median', 'std', 'sem'] or callable
        for key, value in aggregate_funcs.items():
            # Check structure of value
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("Each value in aggregate_funcs must be a tuple of "
                                 f"(column_name, agg_func), got {value} for key {key}.")
            # Check 1)
            if value[0] not in data.columns:
                raise ValueError(f"Column {value[0]} in aggregate_funcs not found in data columns.")
            # Check 2)
            if not isinstance(value[0], str):
                raise ValueError("Column name in aggregate_funcs must be a string,"
                                 f"got {type(value[0])} for key {key}.")
    else:
        raise TypeError("aggregate_funcs must be None, or dict 'feature name':",
                        f"(col_name,func), not {type(aggregate_funcs)}")
    
    # Validation: nonaggregate_funcs, which take the whole pd.DataFrame.GroupBy object and the time bounds
    if nonaggregate_funcs is None:
        # Case of empty dict is already managed by the for loop of nonaggregate_funcs
        nonaggregate_funcs = {} 
    elif isinstance(nonaggregate_funcs, dict):
        # Check if all values in nonaggregate_funcs are callable
        for key, value in nonaggregate_funcs.items():
            if not callable(value):
                raise ValueError("Each value in nonaggregate_funcs must be a "
                                 f"callable, got {value} for key {key}.")
    else:
        raise TypeError("nonaggregate_funcs must be None or dict 'feature name':",
                        f"func, not {type(nonaggregate_funcs)}")

    # Validation: time_bounds must be a tuple of float or None, if None take min-max of bound_window_on
    if time_bounds is None:
        time_bounds = (data[bound_window_on[0]].min(), data[bound_window_on[0]].max())
    elif isinstance(time_bounds, tuple):
        if len(time_bounds) != 2 or not all(isinstance(t, float | int) for t in time_bounds):
            raise ValueError(f"time_bounds must be a tuple of 2 floats or ints, got {time_bounds}.")
    else:
        raise TypeError(f"time_bounds must be None or tuple of 2 floats, not {type(time_bounds)}")

    # Setup
    df = data.copy()
    print(f"Summary for {window_name} with bounds {time_bounds}")
    df_window = []
    for sub in df['Subject'].unique():
        for cond in df['Condition'].unique():
            df_subcond = df.query("Subject == @sub and Condition == @cond")
            first_time = df_subcond[first_time_col].iloc[0] if first_time_col else 0
            lower_bound = first_time + time_bounds[0]
            upper_bound = first_time + time_bounds[1]
            df_filtered = df_subcond.query("@lower_bound <= " + bound_window_on[0] + " <= @upper_bound")
            df_window.append(df_filtered)
    df_window = pd.concat(df_window, ignore_index=True)
    
    if df_window.empty:
        raise ValueError(f"No data found for {window_name} with bounds {time_bounds}.")
    
    # Grouping 
    grouped = df_window.groupby(groupby_columns)
    
    # Direct aggregation
    df_agg = grouped.agg(**aggregate_funcs).reset_index()
    # Fill missing values in the aggregated DataFrame
    # Create a default dataframe with all combinations of groupby columns 
    # to be merged with the grouped DataFrame. This nan-fills all the 
    # combinations of groupby columns expected in the final result
    all_combinations_per_group = []
    from itertools import product
    for group in df_agg['Group'].unique():
        # Get unique values for each groupby column
        groupby_uniques = {col: df_agg[df_agg['Group'] == group][col].unique() 
                           for col in groupby_columns if col != 'Group'}
        # Generate all combinations of unique values, for each group
        all_combinations_this_group = pd.DataFrame(list(product(*groupby_uniques.values())), 
                                                        columns=groupby_uniques.keys())
        all_combinations_this_group['Group'] = group
        all_combinations_per_group.append(all_combinations_this_group)
        
    all_combinations = pd.concat(all_combinations_per_group, ignore_index=True)
    # Window is always the same, so we can add it here
    all_combinations['Window'] = window_name
    # Merge the grouped results with all possible combinations
    merged = pd.merge(all_combinations, df_agg, on=groupby_columns, how='left')
    
    # Indirect aggregation (requiring more than aggregation on one column)
    # these functions are applied to the whole groupby object and are meant
    # to return the dataframe with groupby cols and a new column, with the new
    # feature computed for each group; each time we compute one, we merge
    # to the grouped DataFrame on the groupby columns
    for func_name, func in nonaggregate_funcs.items():
        this_df_nonagg = func(grouped, *time_bounds)
        this_df_nonagg.rename(columns={this_df_nonagg.columns[-1]: func_name}, inplace=True)
        if not isinstance(this_df_nonagg, pd.DataFrame):
            raise ValueError(f"Function {func_name} must return a DataFrame, got {type(this_df_nonagg)}.")
        # Merge
        merged = pd.merge(merged, this_df_nonagg, 
                          on=groupby_columns, how='left', 
                          suffixes=('', f'_{func_name}')
                          )
    
    return merged



def export_significant_html_table(
    df: pd.DataFrame,
    out_path: str,
    vlines: list[int] = [],
    hlines: list[int] = [],
) -> None:
    """
    Esporta HTML con linee personalizzate:
      - hlines_bold: row indices (1-based) per linee orizzontali spesse
      - hlines: row indices (1-based) per linee orizzontali sottili
      - vlines_bold: col indices (1-based) per linee verticali spesse
      - vlines: col indices (1-based) per linee verticali sottili

    Il risultato non mostra né l'indice né le intestazioni originali.
    """
    # de-sparsify
    r_levels = len(df.index.levels) if isinstance(df.index, pd.MultiIndex) else 1
    c_levels = len(df.columns.levels) if isinstance(df.columns, pd.MultiIndex) else 1
    flat = df.reset_index().T.reset_index().T
    flat.iloc[0:r_levels, 0:c_levels] = ''

    # html conversion, this time we have no index or header (de-sparsify)
    table_html = flat.to_html(
        index=False,
        header=False,
        escape=False,
        border=0,
        classes="dataframe"
    )

    # css rule(r)s
    rules = ["table.dataframe { border-collapse: collapse; }"]
    hlines_bold = [c_levels,flat.shape[0]]  # Indici delle righe con hline spessa
    vlines_bold = []  # Indici delle colonne con vline spessa
    if r_levels not in vlines and r_levels not in vlines_bold:
        vlines.insert(0,r_levels)
    # hlines_bold 
    for idx in hlines_bold:
        rules.append(
            f"table.dataframe tr:nth-child({idx}) td "
            "{border-bottom: 3px solid black; }"
        )
    # Also add rule for top solid line
    rules.append(
            "table.dataframe tr:nth-child(1) td "
            "{border-top: 3px solid black; }")
    # hlines 
    for idx in hlines:
        rules.append(
            f"table.dataframe tr:nth-child({idx}) td "
            "{border-bottom: 1px solid black;}"
        )
    # vlines_bold 
    for idx in vlines_bold:
        col = idx
        rules.append(
            f"table.dataframe td:nth-child({col}) "
            "{border-right: 3px solid black;}"
        )
    # vlines 
    for idx in vlines:
        col = idx
        rules.append(
            f"table.dataframe td:nth-child({col}) "
            "{border-right: 1px solid black;}"
        )

    css = "<style type='text/css'>\n" + "\n".join(rules) + "\n</style>\n"

    # css inject
    if "<table" in table_html:
        head, tail = table_html.split("<table", 1)
        full_html = head + css + "<table" + tail
    else:
        warnings.warn("Tabella non trovata nell'HTML generato.", stacklevel=2)
        full_html = css + table_html

    # write
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    return full_html
    
def annotate_panels(axes,
                    xy=(-0.25, 1.05), 
                    xycoords='axes fraction',
                    fontsize=14, 
                    weight='bold',
                    color='black', 
                    ha='left', 
                    va='top'):
    
    for i, ax in enumerate(np.array(axes).flatten()):
        ax.annotate(chr(ord('A') + i), 
                    xy=xy,
                    xycoords=xycoords,
                    fontsize=fontsize,
                    weight=weight,
                    color=color,
                    ha=ha,
                    va=va,)

def get_palette(palette,grp,cnd,default):
    return palette.get(grp,default).get(cnd,default)
            
use_violin = 0
plot_swarm = 0
def boxorviolinplot(x=None, y=None, 
                    data=None, hue=None,
                    palette={}, ax=None,
                    **kwargs):
    
    
            
    if use_violin:
        for nonvalidarg in ['showfliers','flierprops']:
            if nonvalidarg in kwargs:
                kwargs.pop(nonvalidarg)
            
        ax = sns.violinplot(x=x, y=y, data=data, hue=hue,
                       ax=ax,cut=2,
                       inner=None, linewidth=1.25,
                       common_norm=False, 
                       **kwargs)
        
        if plot_swarm:
            sns.swarmplot(x=x, y=y, data=data, hue=hue, dodge=True,
                          alpha=0.75, size=2, 
                          ax=ax,**kwargs)
            from matplotlib.collections import PathCollection
            for coll in [c for c in ax.collections if isinstance(c, PathCollection)]:
                coll.set_facecolor("#7F7F7F")
                coll.set_edgecolor("white")
        
        from matplotlib.collections import PolyCollection
        vpatches = ax.findobj(PolyCollection)
        #print(f"Violin patches found: {len(vpatches)}")
        
        if len(vpatches) == 0:
            warnings.warn("Violin patch color hack (boxorviolinplot): No violin patches found.", stacklevel=2)
        elif len(vpatches) == 3:
            # test whether the hue is present, should be only one, ow warn
            if hue is None or len(data[hue].unique()) != 1:
                warnings.warn(f"Violin patch color hack (boxorviolinplot): Expected one unique hue value, found {len(data[hue].unique())} for hue '{hue}'.")
            
            # Assume two violins, one for healthy and one for patient
            vpatches[0].set_facecolor(get_palette(palette,'HC', 'Resting', 
                                                  vpatches[0].get_facecolor()))
            vpatches[1].set_facecolor(get_palette(palette,'eMCS', 'Resting', 
                                                  vpatches[1].get_facecolor()))
            vpatches[2].set_facecolor(get_palette(palette,'pDoC', 'Resting', 
                                                  vpatches[2].get_facecolor()))
        elif len(vpatches) == 6:
            # test whether the hue is present, should be two, ow warn
            if hue is None or len(data[hue].unique()) != 2:
                warnings.warn(f"Violin patch color hack (boxorviolinplot): Expected two unique hue values, found {len(data[hue].unique())} for hue '{hue}'.")
            
            # Assume four violins, two for healthy and two for patient
            vpatches[0].set_facecolor(get_palette(palette,'HC', 'Resting', 
                                                  vpatches[0].get_facecolor()))
            vpatches[1].set_facecolor(get_palette(palette,'HC', 'Oddball', 
                                                  vpatches[1].get_facecolor()))
            vpatches[2].set_facecolor(get_palette(palette,'eMCS', 'Resting', 
                                                  vpatches[2].get_facecolor()))
            vpatches[3].set_facecolor(get_palette(palette,'eMCS', 'Oddball', 
                                                  vpatches[3].get_facecolor()))
            vpatches[4].set_facecolor(get_palette(palette,'pDoC', 'Resting', 
                                                  vpatches[4].get_facecolor()))
            vpatches[5].set_facecolor(get_palette(palette,'pDoC', 'Oddball', 
                                                  vpatches[5].get_facecolor()))
        else:
            warnings.warn(f"Violin patch color hack (boxorviolinplot): Unexpected number of violin patches: {len(vpatches)}. Expected 2 or 4.")
            
        
    else:
        sns.boxplot(x=x, y=y, data=data, hue=hue,
                    ax=ax, showfliers=False,
                    flierprops=dict(marker='d', markersize=4, 
                                    color='#232323', 
                                    alpha=0.75),
                    linecolor='#343434', 
                    whis=(5, 95),
                    **kwargs)
        
        # adjust box colors in the boxplot branch similar to the violin hack
        boxes = ax.patches
        #print(f"Box patches found: {len(boxes)}")
        if not boxes:
            warnings.warn("Box patch color hack (boxorviolinplot): No box patches found.")
        elif len(boxes) == 3:
            unique_hue = data[hue].unique() if hue is not None else [None]
            if hue is None or len(unique_hue) != 1:
                warnings.warn(f"Box patch color hack (boxorviolinplot): Expected one unique hue value, found {len(unique_hue)} for hue '{hue}'.")
            
            boxes[0].set_facecolor(get_palette(palette,'HC', 'Resting', 
                                                  vpatches[0].get_facecolor()))
            boxes[1].set_facecolor(get_palette(palette,'eMCS', 'Resting', 
                                                  boxes[1].get_facecolor()))
            boxes[2].set_facecolor(get_palette(palette,'pDoC', 'Resting', 
                                                  boxes[2].get_facecolor()))
        elif len(boxes) == 6:
            unique_hue = data[hue].unique() if hue is not None else [None, None]
            if hue is None or len(unique_hue) != 2:
                warnings.warn(f"Box patch color hack (boxorviolinplot): Expected two unique hue values, found {len(unique_hue)} for hue '{hue}'.")
            
            # Assign colors: boxplot goes x first hue second!
            boxes[0].set_facecolor(get_palette(palette,'HC', 'Resting',
                                               boxes[0].get_facecolor()))
            boxes[1].set_facecolor(get_palette(palette,'eMCS', 'Resting',
                                               boxes[2].get_facecolor()))
            boxes[2].set_facecolor(get_palette(palette,'pDoC', 'Resting',
                                               boxes[4].get_facecolor()))
            boxes[3].set_facecolor(get_palette(palette,'HC', 'Oddball',
                                               boxes[1].get_facecolor()))
            boxes[4].set_facecolor(get_palette(palette,'eMCS', 'Oddball',
                                               boxes[3].get_facecolor()))
            boxes[5].set_facecolor(get_palette(palette,'pDoC', 'Oddball',
                                               boxes[5].get_facecolor()))
        else:
            warnings.warn(f"Box patch color hack (boxorviolinplot): Unexpected number of box patches: {len(boxes)}. Expected 3 or 6.")

def get_recording_duration(edf_folder):
    from mne.io import read_raw_edf

    records = []
    all_files = glob(os.path.join(edf_folder, 'sub-*/eog/sub-*.edf'))
    print(f"Found {len(all_files)} EDF files in {edf_folder}")
    for edf_file in all_files:
        if edf_file.lower().endswith(".edf"):
            # Get max recording time
            raw = read_raw_edf(edf_file, preload=False, verbose='ERROR')
            max_time = raw.times[-1] - raw.times[0]
            # Hardcoded parsing for Group, Condition, Subject from fname
            fname = os.path.splitext(os.path.basename(edf_file))[0]
            sub, condition = fname.split('_')[:2]
            if 'sub-' in sub:
                sub = sub.replace('sub-','')
            else:
                raise ValueError(f" 'sub-' not found in filename {fname}")
            if 'task-' in condition:
                condition = condition.replace('task-','')
            rec = {
                "Subject": sub,
                "Condition": condition.capitalize(),
                "max_time": max_time
            }
            records.append(rec)
        else:
            # probably a useless check to be fair, but makes me feel safe rn
            raise ValueError(f"File {edf_file} is not a valid .EDF/.edf file.") 
        
    return pd.DataFrame(records)



def libiv(df,start_time=None,end_time=None) -> float:
    # time bounds not needed, we use blink blink series
    bb = df['BB'].copy().dropna()
    return np.std(np.log(bb/60),ddof=1) if len(bb) > 1 else np.nan


def plot_features_summary(csv_path, axes_layout,palette={}):
    """
    Plot summary using seaborn with a custom axes layout.

    Parameters:
    - csv_path: str or pd.DataFrame
        Path to the CSV file or a pandas DataFrame containing the data.
    - axes_layout: list of tuples
        A 2-D array-like structure where each tuple represents (window, feature) to plot.

    Returns:
    - fig: matplotlib.figure.Figure
        The generated figure.
    """
    # Load CSV
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path
    else:
        df = pd.read_csv(csv_path)

    # Determine layout dimensions
    n_rows = max(1,len(axes_layout))
    n_cols = max(1,max(len(row) for row in axes_layout))
    

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.7, 2 * n_rows), sharex=False)
    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1:
            # If axes is a 1D array, reshape it to 2D
            axes = axes.reshape((n_rows, n_cols))
    else:
        axes = np.array([[axes]])  # Ensure axes is 2D even if only one subplot
    

    # Melt: long-form for seaborn
    features_to_plot = [c for c in df.columns if c not in ['Group', 'Condition', 'Subject']]
    melted = df.melt(
        id_vars=['Group', 'Condition', 'Subject'],
        value_vars=features_to_plot,
        var_name='Feature',
        value_name='Value'
    )

    # Plot each (window, feature) pair in the specified layout
    for i, row in enumerate(axes_layout):
        for j, (window, feature) in enumerate(row):
            if window is None or feature is None:
                axes[i, j].axis('off')  # Turn off unused axes
                continue

            #print(f'Plotting {feature}-{window}')
            boxorviolinplot(
                data=melted.query("Feature == @feature"),
                x='Group',
                y='Value',
                hue='Condition',
                ax=axes[i, j],
                legend=False,
                palette=palette
            )

            # Set labels and titles
            axes[i, j].set_ylabel(f"{feature}".replace('_', '\n'), fontsize=10)

            if i == 0:
                axes[i, j].set_title(window, fontsize=10)
            
            if i != n_rows - 1:
                axes[i, j].set_xlabel('')
                axes[i, j].set_xticklabels([])
                
            

    plt.tight_layout(h_pad=0.1, w_pad=0.001)
    return fig

# Stats
def run_stat_tests_ncheck(results, 
                   condition_col='Condition', 
                   group_col='Group', 
                   alpha=0.05, 
                   p_adjust='holm',
                   group_comparisons=[('HC', 'EMCS'), ('EMCS', 'DoC'), ('HC', 'DoC')]):
    
    def fmt_p(p):
        try: p = float(p)
        except: return p
        return (
            f"{p:.3f}***" if p < 0.001 else
            f"{p:.3f}**" if p < 0.01 else
            f"{p:.3f}*"  if p < 0.05 else
            f"{p:.3f}\u25C6" if p < 0.10 else
            f"{p:.3f}"
        )
    stats_table = []
    # Loop through all numeric features
    for feature in [f for f in results.columns if results[f].dtype == float or results[f].dtype == int]:
        for cond in results[condition_col].unique():
            # Filter data for this condition
            cond_data = results[results[condition_col] == cond]

            # Perform normality test (Shapiro-Wilk) for each group in the condition
            row = {('Feature', 'Condition'): (feature.split('[')[0].strip(), cond)}
            for grp in cond_data[group_col].unique():
                grp_data = cond_data[cond_data[group_col] == grp][feature].dropna()
                if len(grp_data) >= 3:
                    _, p_val = stats.shapiro(grp_data)
                else:
                    p_val = None
                mdn, q1, q3 = np.median(grp_data), np.percentile(grp_data, 25), np.percentile(grp_data, 75)
                row[("Median (IQR)",f"{grp}")] = f"{mdn:.1f} ({q1:.1f}, {q3:.1f}){'◆' if p_val is not None and p_val < 0.05 else ''}"
                row[('Normality Test', f"{grp} p")] = p_val

            # Gather values for each group
            groups_data = []
            for grp in cond_data[group_col].unique():
                grp_values = cond_data[cond_data[group_col] == grp][feature].dropna()
                if len(grp_values) > 0:
                    groups_data.append(grp_values)

            if len(groups_data) >= 2:
                    
                h_stat, p_val = stats.kruskal(*groups_data)
                row.update({
                    ('Kruskal-Wallis', 'H (df=2)'): h_stat,
                    ('Kruskal-Wallis', 'p'): fmt_p(p_val),
                    ('Kruskal-Wallis', 'η²'): h_stat / (len(cond_data) - 1) if len(cond_data) > 1 else None
                })

                if p_val < alpha:
                    posthoc_data = cond_data[[group_col, feature]].dropna()
                    if len(posthoc_data) > 0:
                        try:
                            posthoc_result = skph.posthoc_conover(posthoc_data,
                                                                  val_col=feature,
                                                                  group_col=group_col,
                                                                  p_adjust=None)
                            posthoc_result_adjust = skph.posthoc_conover(posthoc_data,
                                                                    val_col=feature,
                                                                    group_col=group_col,
                                                                    p_adjust=p_adjust)
                                
                            for group1, group2 in group_comparisons:
                                # Standardized column names for both cases
                                if group1 in posthoc_result.index and group2 in posthoc_result.columns:
                                    row[(f'Conover', f'{group1} vs. {group2}')] = ''
                                    p = posthoc_result.loc[group1, group2]
                                    padj = posthoc_result_adjust.loc[group1, group2]
                                    p_ = fmt_p(p)
                                    padj_ = fmt_p(padj)
                                    row[(f'Conover', f'{group1} vs. {group2}')] = f"{p_} ({padj_})"
                                    
                        except Exception as e:
                            warnings.warn(f"Post-hoc test Conover failed for feature {feature} and condition {cond}: {e}")
                else:
                        for group1, group2 in group_comparisons:
                            row[(f'Conover', f'{group1} vs. {group2}')] = ''
            else:
                # If fewer than two groups, add empty KW test and fields
                row.update({
                    ('Kruskal-Wallis', 'H (df=2)'): None,
                    ('Kruskal-Wallis', 'p'): None,
                    ('Kruskal-Wallis', 'η²'): None
                })
            stats_table.append(row)
            
    # Build the results DataFrame with a multi-index
    results_df = pd.DataFrame(stats_table)
    results_df.set_index([('Feature', 'Condition')], inplace=True)
    results_df.index = pd.MultiIndex.from_tuples(results_df.index)
    results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)
    return results_df

def run_stat_tests_wilcoxon(df, split_by='Group', pair_by='Condition'):
    """
    For each numeric feature, within each level of split_by, perform pairwise
    Wilcoxon signed-rank tests between levels of pair_by.
    
    Returns a DataFrame with index as Feature and columns as a MultiIndex
    Level 0: Group
    Level 1: 'W' & 'p'.
    """
    


    stats_table = []
    # Identify numeric features.
    numeric_features = [f for f in df.columns if df[f].dtype in [float, int]]
    
    for feature in numeric_features:
        # Create a dictionary with explicit Feature and Group columns.
        row = {'Feature': feature.split('[')[0].strip()}
        for grp in df[split_by].unique():
            subset = df[df[split_by] == grp]
            conditions = sorted(subset[pair_by].unique())
            
            for cond1, cond2 in itertools.combinations(conditions, 2):
                # Extract paired observations using Subject as key.
                df1 = subset[subset[pair_by] == cond1][['Subject', feature]].dropna()
                df2 = subset[subset[pair_by] == cond2][['Subject', feature]].dropna()
                merged = pd.merge(df1, df2, on='Subject', suffixes=('_' + cond1, '_' + cond2))
                if len(merged) >= 1:
                    try:
                        data1 = merged[f'{feature}_{cond1}']
                        data2 = merged[f'{feature}_{cond2}']
                        str1 = f'{cond1}: <{np.quantile(data1,0.25):.1f}|{np.median(data1):.1f}|{np.quantile(data1,0.75):.1f}>'
                        str2 = f'{cond2}: <{np.quantile(data2,0.25):.1f}|{np.median(data2):.1f}|{np.quantile(data2,0.75):.1f}>'
                        print(f"Comparing  {grp} on {feature}: {str1}-{str2}")
                        stat, p_val = stats.wilcoxon(data1,data2)
                        print(f"Resulting: W={stat},p={p_val}")
                    except Exception:
                        stat, p_val = None, None
                else:
                    stat, p_val = None, None
                row[(f'{grp}: {cond1}-{cond2}','W')] = stat
                row[(f'{grp}: {cond1}-{cond2}','p')] = p_val
        stats_table.append(row)
    
    # Build DataFrame; note that column keys that are tuples remain as is.
    df = pd.DataFrame(stats_table)
    
    # set feature as index
    df = df.set_index(["Feature"])
    
    # set multi-index columns with one level for group and one for statistics
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df

def _r2_cs_nagelkerke(model):
    n   = model.nobs
    ll0 = model.llnull  # log-likelihood null model
    ll1 = model.llf     # log-likelihood fitted model
    r2_cs = 1 - np.exp((2.0/n)*(ll0 - ll1)) # Cox-Snell
    # Cox-Snell is bounded by 1 - L0^(2/n) 
    r2_n  = r2_cs / (1 - np.exp((2.0/n)*ll0)) # Nagelkerke
    return r2_cs, r2_n

def logreg_covariate(
        df: pd.DataFrame,
        feat: str,
        covariate: Union[str, List[str]],
        label_col: str,
        get_patients_binary: callable = lambda series: (series != 'HC').astype(int).values,
        verbose: bool = True,
        standardize: bool = True,
        plot = False
    ) -> pd.DataFrame:
        
        def fmt_p(p):
            try: 
                p = float(p)
            except: 
                return p
            return (
                f"{p:.3f}***" if p < 0.001 else
                f"{p:.3f}**" if p < 0.01 else
                f"{p:.3f}*"  if p < 0.05 else
                f"{p:.3f}\u25C6" if p < 0.10 else
                f"{p:.3f}"
            )
        
        if isinstance(covariate, str):
            covariate = [covariate]
        elif not (isinstance(covariate, list) 
                and all(isinstance(cov, str) 
                        for cov in covariate)
                ):
            raise ValueError("covariate must be a string or a list of strings.")
        
        # Restrict to feature, covariate(s), and label_col
        df_clean = df.copy()[[feat] + covariate + [label_col]]
        
        # If standardize, standardize the feature and covariates
        if standardize:
            from sklearn.preprocessing import StandardScaler

            
            scaler = StandardScaler()
            df_clean[[feat] + covariate] = scaler.fit_transform(df_clean[[feat] + covariate])
            
        else:
            warnings.warn("Standardization is set to False, which may affect the model performance. Consider standardizing the feature and covariate for better results.")
        
        # Warn if some rows were dropped due to NaN values in feature, covariate(s), or label_col
        if df_clean.shape[0] != df.shape[0]:
            warnings.warn(f"{df.shape[0] - df_clean.shape[0]} rows dropped due to NaN values in feature, covariate(s), or label_col.")
        
        # Binary label: 0 = HC, 1 = Patient (or as defined by get_patients_binary)
        if not callable(get_patients_binary):
            raise ValueError("get_patients_binary must be a callable that returns a binary Series.")
        else:
            y = get_patients_binary(df[label_col])
        
        # Univariate logistic regression model for the feature
        X_univ = sm.add_constant(df_clean[[feat]])
        model_univ = sm.Logit(y, X_univ).fit(disp=False)
        if verbose:
            print(f"\nUnivariate model for feature '{feat}' fitted.")
            print(f"Model summary:\n{model_univ.summary()}")
        
        # Multivariate logistic regression model with feature and covariates
        X_mult = sm.add_constant(df_clean[[feat] + covariate])
        model_mult = sm.Logit(y, X_mult).fit(disp=False)
        if verbose:
            print(f"\nMultivariate model for feature '{feat}' with covariate(s) '{covariate}' fitted.")
            print(f"Model summary:\n{model_mult.summary()}")
            
        # For both models compute AUC and Nagelkerke pseudo R2
        # Univariate predictions and metrics
        y_pred_univ = model_univ.predict(X_univ)
        auc_univ = roc_auc_score(y, y_pred_univ)
        r2_cs_univ, r2_nagelkerke_univ = _r2_cs_nagelkerke(model_univ)

        # Multivariate predictions and metrics
        y_pred_mult = model_mult.predict(X_mult)
        auc_mult = roc_auc_score(y, y_pred_mult)
        r2_cs_mult, r2_nagelkerke_mult = _r2_cs_nagelkerke(model_mult)
        
        if plot:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 5))
            if len(covariate) == 1:
                plt.scatter(df_clean[feat], df_clean[covariate[0]], c=y, cmap='coolwarm', alpha=0.7)
                plt.xlabel(feat)
                plt.ylabel(covariate[0])
                plt.title(f"{feat} vs {covariate[0]}")
            elif len(covariate) == 2:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df_clean[feat], df_clean[covariate[0]], df_clean[covariate[1]], c=y, cmap='coolwarm', alpha=0.7)
                ax.set_xlabel(feat)
                ax.set_ylabel(covariate[0])
                ax.set_zlabel(covariate[1])
                ax.set_title(f"{feat} vs {covariate[0]} vs {covariate[1]}")
            else:
                warnings.warn("Plotting only supported for up to 2 covariates.")
            row_text = (
                f"OR={np.exp(model_univ.params[feat]):.2f}, p={fmt_p(model_univ.pvalues[feat])}\n"
                f"Corr. OR={np.exp(model_mult.params[feat]):.2f}, p={fmt_p(model_mult.pvalues[feat])}\n"
                f"Corr. Nagelkerke R2={r2_nagelkerke_mult:.2f}\n",
                f"Corr. AUC={auc_mult:.2f}"
            )
            plt.gcf().text(0.02, 0.98, row_text, fontsize=9, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
            plt.tight_layout()
            plt.show(block=False)
        
        return pd.DataFrame({
                # univariate model results
                'feat_OR': round(np.exp(model_univ.params[feat]), 3),
                'feat_CI_lower': round(np.exp(model_univ.conf_int().loc[feat, 0]), 3),
                'feat_CI_upper': round(np.exp(model_univ.conf_int().loc[feat, 1]), 3),
                'feat_p-value': fmt_p(model_univ.pvalues[feat]),
                'feat_AUC': round(auc_univ, 3),
                'feat_R2_CoxSnell': round(r2_cs_univ, 3),
                'feat_R2_Nagelkerke': round(r2_nagelkerke_univ, 3),
                # multivariate model results
                'feat_corrected_OR': round(np.exp(model_mult.params[feat]), 3),
                'feat_corrected_CI_lower': round(np.exp(model_mult.conf_int().loc[feat, 0]), 3),
                'feat_corrected_CI_upper': round(np.exp(model_mult.conf_int().loc[feat, 1]), 3),
                'feat_corrected_p-value': fmt_p(model_mult.pvalues[feat]),
                'feat_corrected_AUC': round(auc_mult, 3),
                'feat_corrected_R2_CoxSnell': round(r2_cs_mult, 3),
                'feat_corrected_R2_Nagelkerke': round(r2_nagelkerke_mult, 3),
                
                # stuff used for standardization check
                'feat_mean': round(df.copy()[feat].mean(), 3),
                'feat_std': round(df.copy()[feat].std(), 3),
                **{f'covariate_{cov}_mean': round(df.copy()[cov].mean(), 3) 
                   for cov in covariate},
                **{f'covariate_{cov}_std': round(df.copy()[cov].std(), 3) 
                   for cov in covariate},
                
                # multivariate model params (all)
                **{f'param_{param}': round(value, 3) for param, value in model_mult.params.items()},
                
                # multivariate model covariates results
                **{f'covariate_{cov}_CI_lower': round(np.exp(model_mult.conf_int().loc[cov, 0]), 3) 
                   for cov in covariate},
                **{f'covariate_{cov}_CI_upper': round(np.exp(model_mult.conf_int().loc[cov, 1]), 3) 
                   for cov in covariate},
                **{f'covariate_{cov}_p-value': fmt_p(model_mult.pvalues[cov]) 
                   for cov in covariate},
                

            }, index=[feat.split('[')[0].strip()]  # Use the feature name without units as index
        )


def plot_logreg_effects(res, feat, covariates, group_col, data, group2col=None, show_prob=False, 
                        get_patients_binary: callable = lambda series: (series != 'HC').astype(int).values):
    """
    Visualize logistic regression effects as either multiplicative odds or marginal probabilities.

    Parameters
    ----------
    res : pd.DataFrame
        Output from logreg_covariate().
    feat : str
        Main feature name.
    covariates : list[str]
        Covariate names.
    group_col : str
        Column name for binary class.
    data : pd.DataFrame
        DataFrame with feat, covariates, and group_col.
    group2col : dict, optional
        Mapping of group labels -> colors. If None, all points black.
    show_prob : bool, default=False
        If True, plot marginal probability curve (others fixed at mean),
        with actual class values (0-1) as scatter and not relative odds.
    get_patients_binary : callable, optional
        Function to convert group_col to binary (0/1). Default maps 'HC' to 0, others to 1.
    """

    # Extract coefficients, means, stds
    coefs = {'const': res['param_const'].iloc[0]}
    means, stds = {}, {}

    coefs[feat] = res[f'param_{feat}'].iloc[0]
    means[feat] = res['feat_mean'].iloc[0]
    stds[feat]  = res['feat_std'].iloc[0]

    for cov in covariates:
        coefs[cov] = res[f'param_{cov}'].iloc[0]
        means[cov] = res[f'covariate_{cov}_mean'].iloc[0]
        stds[cov]  = res[f'covariate_{cov}_std'].iloc[0]

    all_vars = [feat] + covariates
    n = len(all_vars)

    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), 
                             sharey=not show_prob) # only for relative odds
    if n == 1:
        axes = [axes]
        
    # compute actual ymin, ymax of relative odds if not show_prob, o.w. None
    ylims = None
    if not show_prob:
        for var in all_vars:
            beta = coefs[var]
            mu, sigma = means[var], stds[var]
            const = coefs['const']

            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)
            z = (x - mu) / sigma

            y = np.exp(beta * z)
            y_data = np.exp(beta * (data[var] - mu) / sigma)

            y_min = min(np.min(y), np.min(y_data))
            y_max = max(np.max(y), np.max(y_data))

            if ylims is None:
                ylims = [y_min, y_max]
            else:
                ylims[0] = min(ylims[0], y_min)
                ylims[1] = max(ylims[1], y_max)

    for ax, var in zip(axes, all_vars, strict=False):
        beta = coefs[var]
        mu, sigma = means[var], stds[var]
        const = coefs['const']

        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)
        z = (x - mu) / sigma

        # Marginal probability or odds curve
        if show_prob:
            logit = const + beta * z
            y = 1 / (1 + np.exp(-logit))
            ylabel = "Marginal probability"
        else:
            y = np.exp(beta * z)
            ylabel = "Relative odds (multiplicative)"

        # Shaded background
        for k in range(1, 4):
            ax.axvspan(mu - k*sigma, mu + k*sigma, color='C0', alpha=0.15/k)
        ax.axvline(mu, color='k', ls='--', lw=1)

        # Curve
        ax.plot(x, y, color='C0', lw=2, label='Model')

        # Scatter points
        colors = data[group_col].map(group2col) if group2col is not None else ['black'] * len(data)

        if show_prob:
            ax.scatter(
                data[var], data[group_col],
                c=colors, edgecolor='k', alpha=0.7, label='Samples'
            )
        else:
            y_data = np.exp(beta * (data[var] - mu) / sigma)
            y_min, y_max = ylims # with sharey we need to consider global ylims
            disp = 0.05 * (y_max - y_min)
            y_data_disp = np.where(get_patients_binary(data[group_col]) == 1, y_data + disp, y_data - disp)
            ax.scatter(
                data[var], y_data_disp,
                c=colors, edgecolor='k', alpha=0.7, label='Samples'
            )

        ax.set_title(var + (" (rel. odds)" if not show_prob else " (marg. prob.)"))
        ax.set_xlabel(f"{var} (natural units)")
        ax.set_ylabel(ylabel)

        # --- Annotation of OR, CI, p-value
        if var == feat:
            # for feat, the ones with no "corrected" are univariate
            OR = res["feat_corrected_OR"].iloc[0]
            CI_l = res["feat_corrected_CI_lower"].iloc[0]
            CI_u = res["feat_corrected_CI_upper"].iloc[0]
            pval = res["feat_corrected_p-value"].iloc[0]
        else:
            OR = np.exp(coefs[var])
            try:
                CI_l = res[f'covariate_{var}_CI_lower'].iloc[0]
                CI_u = res[f'covariate_{var}_CI_upper'].iloc[0]
            except Exception as e:
                print(f"Warning: Could not retrieve CI for covariate {var}: {e}")
                CI_l, CI_u = np.nan, np.nan
            try:
                pval = res[f'covariate_{var}_p-value'].iloc[0]
            except Exception as e:
                print(f"Warning: Could not retrieve p-value for covariate {var}: {e}")
                pval = np.nan

        annotation_text = f"OR = {OR:.3f}\n95% CI = [{CI_l:.3f}, {CI_u:.3f}]\np = {pval}"

        # Temporarily create legend to obtain bbox
        leg = ax.legend(loc="best")
        fig.canvas.draw()
        bbox = leg.get_window_extent()
        bbox_data = bbox.transformed(ax.transAxes.inverted())
        leg.remove()  # remove legend

        # Coordinates in axes fraction units
        x0, y0, width, height = bbox_data.x0, bbox_data.y0, bbox_data.width, bbox_data.height
        x_text, y_text = x0 + width * 0.05, y0 + height * 0.5

        ax.text(
            x_text, y_text, annotation_text,
            transform=ax.transAxes,
            fontsize=9, va="center", ha="left",
            bbox=dict(facecolor='white', alpha=0.30, edgecolor='none')
        )

    plt.tight_layout()
    return fig