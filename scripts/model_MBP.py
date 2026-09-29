# %%
NBIN = 10
OVBIN = 0 # Overlap size for MBP computation, as proportion of bin width
INTP_DT = 0.01  # with low overlap, used to stabilize fitting
# MBP := mean blink proportion in bins of size WBIN seconds
# MBP_0 := theoretical MBP for random blinking 
TMIN = -0.5
TMAX = 0.5
C_HC, C_EMCS, C_PDOC = '#0072B2', '#009E73', '#D55E00'  # Colors for plotting

# Global visualization and model parameters
T_STIM = 0.050  # seconds
T_STIM_MS = int(T_STIM * 1000)  # milliseconds

ALPHA = 0.05  # significance level
RAYLEIGH_PLOT_RESTING = False  # default: plot only ODDBALL unless explicitly enabled

# stim icon
STIM_ICON = "♫"
STIM_ICON_UTF8 = "\u266B"


# %% [markdown]
# ### Imports 

# %%
# Import of standard libraries
import os
from contextlib import suppress
from glob import glob
from traceback import format_exc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Additional utilities
from IPython.display import display
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm

# Set matplotlib style and backend
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.spines.right': False,
    'axes.spines.top': False
})
#%matplotlib inline

plt.close('all')  # Close all open figures

pd.set_option('display.max_rows',5)
pd.set_option('display.max_columns', 25)  # Show all columns in DataFrame

# Centralized output folder for this script (all generated artifacts)
OUTPUT_DIR = './results_MBP'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def save_table_csv(df, filename, float_format='%.6g'):
    """Save a DataFrame table into OUTPUT_DIR/tables with ';' delimiter."""
    df.to_csv(os.path.join(TABLES_DIR, filename), sep=';', float_format=float_format)

# ---------- Helpers for consistent annotation ----------

def add_stim_to_ax(ax, t0=0.0, t_stim=T_STIM, bar_frac_width=0.08, bar_ypos='top',
                   label='Time from stimulus onset (ms)', bar_color='k', bar_alpha=1.0,
                   icon=STIM_ICON, keep_xlim=True, keep_ylim=True):
    """Annotate axes with stimulus onset vertical line and a black bar of length t_stim.

    - t0: stimulus time in seconds
    - t_stim: duration in seconds
    - bar_frac_width: fraction of y-range used as bar thickness
    - bar_ypos: 'top' or 'bottom' bar placement
    - label: x-axis label text
    - icon: draw a tiny speaker-like glyph near the bar
    - keep_xlim: restore original xlim after plotting
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # vertical line at t0
    ax.axvline(t0, color='k', linestyle='-', linewidth=1.5, alpha=0.9, zorder=10)
    # bar coordinates
    y0, y1 = ylim
    yr = y1 - y0
    thickness = yr * bar_frac_width
    ybar = y1 - thickness * 1.2 if bar_ypos == 'top' else y0 + thickness * 0.2
    ax.hlines(ybar, t0, t0 + t_stim, colors=bar_color, alpha=bar_alpha, linewidth=3, zorder=10)
    if icon:
        # annotate with icon near the bar, suppress any annotation-related exceptions
        with suppress(Exception):
            ax.annotate(icon, xy=(t0 + t_stim/2, ybar + thickness*0.6),
                        xytext=(0, thickness*1.5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=12,
                        fontname='DejaVu Sans')  # font supporting utf-8 icons
            pass
    # axis labels
    ax.set_xlabel('Time from stimulus onset (ms)')
    # Format x-axis from seconds to milliseconds
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x*1000))}"))
    if keep_xlim:
        ax.set_xlim(xlim)
    if keep_ylim:
        ax.set_ylim(ylim)
    return ybar  # return y position of the bar for further annotations if needed
        
def finalize_mbp_axes(ax, hline=None, hband=None):
    ax.set_ylabel('MBP')
    ax.grid(True, which='major', linestyle='-', linewidth=0.75)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5)
    # null values
    if hline:
        ax.axhline(hline, color='gray', linestyle='--', linewidth=2.5, alpha=0.5)
    if hband:
        ax.axhspan(hline-hband, hline+hband, color='gray', alpha=0.2)
        
    #percent formatting
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    return ax

# %% [markdown]
# #### MBP computation utils

# %%

# MBP computation and plotting functions

def null_mbp(B=None, tmin=None, tmax=None, bin_width=None):
    """Return the null MBP value (for random blinking) as per Huber et al. 2022.
       Parametrization with B has priority over bin_width/tmin/tmax.
    """
    if B is None:
        return (tmax - tmin) / bin_width
    else:
        return 1/B


def null_mbp_var(N_blinks, B = None, tmin=None, tmax=None, bin_width=None):
    """Return the variance of the null MBP value.
       Parametrization with B has priority over bin_width/tmin/tmax.
    """
    if B is None:
        B = bin_width / (tmax - tmin)
        assert np.isclose(B, round(B)), f"B must be integer, instead is {B}"
    
    try:
        return (B-1) / (N_blinks * B**2)
    except ZeroDivisionError as zde:
        print(f"Error computing null variance for B = {B} and N_blinks = {N_blinks}:  {zde}")
        return np.nan
    

def compute_counts(stimuli, blinks, *, tmin, tmax, nbin, overlap_ratio=0.0):
    """
    Count blinks per bin.

    Non-overlap (overlap_ratio == 0.0): unchanged, half-open bins tile [tmin, tmax).
    Overlap (overlap_ratio > 0.0): circular; advance by step and wrap starts modulo [tmin, tmax)
    until a wrapped start repeats (modulus repeat). Bins crossing tmax are split at the boundary.

    Returns (times, counts, n):
      - times: bin centers wrapped to [tmin, tmax) and sorted
      - counts: counts per (possibly overlapping) circular bin summed over stimuli
      - n: total blinks in [stimulus+tmin, stimulus+tmax) across stimuli (independent of overlap)
    """
    import numpy as np

    bin_width = (tmax - tmin) / nbin
    overlap_width = bin_width * overlap_ratio

    if bin_width <= 0:
        raise ValueError("bin_width must be > 0")
    if overlap_width < 0:
        raise ValueError("overlap_width must be >= 0")
    step = bin_width - overlap_width
    if step <= 0:
        raise ValueError("overlap_width must be < bin_width")
    if tmax <= tmin:
        raise ValueError("Require tmax > tmin")

    stimuli = np.asarray(stimuli, dtype=float)
    blinks  = np.asarray(blinks,  dtype=float)
    L = (tmax - tmin)

    # --- No-overlap path: keep exactly as-is ---
    if overlap_width == 0.0:
        span = L - bin_width
        if span < -1e-12:
            raise ValueError("bin_width is larger than window length (tmax - tmin)")
        n_steps = int(np.floor(max(span, 0.0) / step)) + 1
        starts  = tmin + step * np.arange(n_steps)
        stops   = starts + bin_width
        times   = starts + 0.5 * bin_width

        counts = np.zeros_like(starts, dtype=float)
        n = 0

        if blinks.size == 0 or stimuli.size == 0:
            return times, counts, n

        blinks_sorted = np.sort(blinks)
        for te in stimuli:
            rel = blinks_sorted - te
            n += np.count_nonzero((rel >= tmin) & (rel < tmax))
            i0 = np.searchsorted(rel, starts, side='left')
            i1 = np.searchsorted(rel, stops,  side='left')
            counts += (i1 - i0)
        return times, counts, n

    # --- Overlap path: circular until modulus repeat ---
    # Generate wrapped starts until a start modulo [tmin, tmax) repeats
    starts_wrapped = []
    seen = set()
    max_iters = int(1e6)  # guard for pathological ratios
    k = 0
    while k < max_iters:
        s  = tmin + k * step
        sw = tmin + np.mod(s - tmin, L)  # wrap to [tmin, tmax)
        key = int(np.round(((sw - tmin) / L) * 1e12))  # quantize to avoid FP drift
        if key in seen:
            break
        seen.add(key)
        starts_wrapped.append(sw)
        k += 1
    if k == max_iters:
        raise RuntimeError("Exceeded maximum iterations while generating circular bins; check parameters.")

    starts  = np.asarray(starts_wrapped, dtype=float)
    stops   = starts + bin_width
    centers = tmin + np.mod(starts + 0.5 * bin_width - tmin, L)

    counts = np.zeros_like(starts, dtype=float)
    n = 0

    if blinks.size == 0 or stimuli.size == 0:
        order = np.argsort(centers)
        return centers[order], counts[order], n

    blinks_sorted = np.sort(blinks)
    for te in stimuli:
        rel = blinks_sorted - te
        in_win = (rel >= tmin) & (rel < tmax)  # do not wrap data; window restriction holds
        relw = rel[in_win]
        n += relw.size
        if relw.size == 0:
            continue

        # Base segment: [start, min(stop, tmax))
        stops_clip = np.minimum(stops, tmax)
        i0 = np.searchsorted(relw, starts,     side='left')
        i1 = np.searchsorted(relw, stops_clip, side='left')
        counts += (i1 - i0)

        # Wrapped tails for bins with stop > tmax: [tmin, stop - L)
        wrap_mask = stops > tmax
        if np.any(wrap_mask):
            tail_stops = tmin + (stops[wrap_mask] - tmax)  # stop - L
            i_wr = np.searchsorted(relw, tail_stops, side='left')  # lower bound is tmin -> index 0
            counts[wrap_mask] += i_wr

    order = np.argsort(centers)
    return centers[order], counts[order], n


    
def compute_mbp(stimuli, blinks, *, tmin, tmax, nbin, overlap_ratio=0.0):
    """
    MBP = counts / n. For overlap_ratio == 0, MBP sums to 1 over bins.
    For overlap_ratio > 0, normalization is still consistent with no-overlap!
    """
    times, counts, n = compute_counts(stimuli, blinks,
                                      tmin=tmin, tmax=tmax,
                                      nbin=nbin, overlap_ratio=overlap_ratio)
    if n == 0:
        return times, np.full_like(counts, np.nan, dtype=float), n
    return times, counts / n, n

def wrap_to_window(x, tmin=TMIN, tmax=TMAX):
    period = tmax - tmin
    return tmin + np.mod(np.asarray(x) - tmin, period)

def _to_phase(rel_t, tmin=TMIN, tmax=TMAX):
    period = (tmax - tmin)
    return 2 * np.pi * (wrap_to_window(rel_t, tmin=tmin, tmax=tmax) - tmin) / period

def rayleigh_test(phases):
    """
    Rayleigh test for non-uniformity on the circle.
    Returns dict with n, rbar, z, p.
    """
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    n = phases.size
    if n == 0:
        return {'n': 0, 'rbar': np.nan, 'z': np.nan, 'p': np.nan}

    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R = np.sqrt(C**2 + S**2)
    rbar = R / n
    z = n * (rbar**2)

    # finite-sample correction (Berens 2009 / CircStat convention)
    p = np.exp(-z) * (
        1
        + (2*z - z**2) / (4*n)
        - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*(n**2))
    )
    p = float(np.clip(p, 0.0, 1.0))
    return {'n': int(n), 'rbar': float(rbar), 'z': float(z), 'p': p}

def _read_col_anycase(df, preferred):
    cols_lower = {c.lower(): c for c in df.columns}
    return df[cols_lower[preferred.lower()]]

def collect_relative_blinks(stimuli, blinks, tmin=TMIN, tmax=TMAX):
    """Collect all blink times relative to each stimulus in [tmin, tmax)."""
    stimuli = np.asarray(stimuli, dtype=float)
    blinks = np.asarray(blinks, dtype=float)
    if stimuli.size == 0 or blinks.size == 0:
        return np.array([], dtype=float)

    rel_all = []
    blinks_sorted = np.sort(blinks)
    for te in stimuli:
        rel = blinks_sorted - te
        mask = (rel >= tmin) & (rel < tmax)
        if np.any(mask):
            rel_all.append(rel[mask])
    if len(rel_all) == 0:
        return np.array([], dtype=float)
    return np.concatenate(rel_all)


# %% [markdown]
# #### MBP fitting and plotting utils

# %%

# MBP fitting utilities. modeling function requires:
# 1) model func: (x, *params) -> y(x; params)
# 2) guess func: (y0, interval) -> initial_params
# 3) bound func: (y0, interval) -> bounds_array of shape (2, n_params)

def create_fitting(data,x,xlim,y,y0,model,guess,bound):

    fdata = data.copy().query(f"{x} >= {xlim[0]} and {x} <= {xlim[1]}")
    yvals = fdata[y].to_numpy()
    xvals = fdata[x].to_numpy()

    try:
        pfit, pcov, info, msg, ier = curve_fit(model, xvals, yvals, 
                                                p0=guess(y0, xlim), 
                                                bounds=bound(y0, xlim),
                                                max_nfev=1000*len(guess(y0, xlim)), # default is 100*n_params
                                                jac='2-point',
                                                full_output=True
                                                )
        res_over_sigma = info['fvec']
        loss = np.sum(res_over_sigma**2) # corresponds to RSS when loss is linear (default) and we gave no sigma (identity sigma)
        pstd = np.sqrt(np.diag(pcov))
        
    except RuntimeError as e:
        print(e)
        print(format_exc())
        pfit = np.array([np.nan]*len(guess(y0, xlim)))
        pcov = np.full((len(pfit),len(pfit)), np.nan)
        pstd = np.array([np.nan]*len(guess(y0, xlim)))
        loss = np.nan
    
    
    return pfit, pstd, loss

def create_fitted_data_same(data, x, xlim, y, model, *pfit):
    fitteddata = data.copy()
    fitteddata[y] = np.nan
    mask = (xlim[0] <= fitteddata[x]) & (fitteddata[x] <= xlim[1])
    fitteddata.loc[mask, y] = model(fitteddata.loc[mask, x].to_numpy(), *pfit)
    maskdata = (fitteddata[x] < min(data[x])) | (fitteddata[x] > max(data[x]))
    fitteddata.loc[maskdata, y] = np.nan # never extrapolate
    return fitteddata

def circ_interp(x, y, xnew):
    x = np.concatenate([[x[0] - (x[1]-x[0])], x, [x[-1] + (x[-1]-x[-2])]])
    y = np.concatenate([[y[-1]], y, [y[0]]])
    ynew = np.interp(xnew, x, y)
    return ynew

def fit_allsubjects(df, subjects, condition, x, y, y0,
                    model, guess, bound, names, 
                    interpolate_dt = None,
                    xmin = TMIN, 
                    xmax = TMAX ):
    pfit_df = []

    for subject in (pbar:=tqdm(subjects,desc='Fitting subjects')):
        pbar.set_description(f'Fitting [{subject}]')
        
        # Filter data for the current subject
        subject_data = df.query(f"Subject == '{subject}' and Condition == '{condition}'")
        
        # y0 may be a float (same for all subjects) or a column name in df
        y0 = y0 if isinstance(y0, float) else subject_data[y0].iloc[0]
        if pd.isna(y0) or np.isnan(y0):
            print(f"Subject {subject} has NaN y0, skipping.")
            continue
        
        # optional interpolation 
        if interpolate_dt is None:
            data2fit = subject_data
        else:
            upsampled_t = np.arange(xmin, xmax, interpolate_dt)
            mbp_linint = circ_interp(
                subject_data[x].to_numpy(),
                subject_data[y].to_numpy(),
                upsampled_t
            )
            data2fit = pd.DataFrame({
                x: upsampled_t,
                y: mbp_linint
            })
            
        # fit
        pfit, pstd, loss = create_fitting(data2fit,x,(xmin,xmax),y,
                                        y0,
                                        model,
                                        guess,
                                        bound,
                                        )
        
        pfit_dict = {p: v for p,v in zip(names, pfit, strict=True)}
        pstd_dict = {f"{p}_std": v for p,v in zip(names, pstd, strict=True)}
        loss_dict = {'loss': loss}
        all_params = {**pfit_dict, **pstd_dict, **loss_dict}
        
        # store pfit
        pfit_df.append(pd.DataFrame(all_params, 
                                    index=[subject]))
        
    pfit_df = pd.concat(pfit_df)
    return pfit_df



# %%
# plotting utilities
def plot_mbpfit_allsubjects(df, pfit_df, subjects, condition, x, y, y0, model, names, 
                            vary0 = None,
                            xmin = TMIN, xmax = TMAX):
    plt.close('all')
    # Create subplots for each subject
    n_subjects = len(subjects)
    fig, axes = plt.subplots(n_subjects, 1, figsize=(12, n_subjects * 2), sharex=True, sharey=True)
    
    if n_subjects == 1:
        axes = [axes]

    for ax,subject in zip(axes,subjects, strict=True):
        subject_data = df.query(f"Subject == '{subject}' and Condition == '{condition}'")
        
        pfit = pfit_df.loc[subject,names].to_numpy()
        
        # y0 may be a float (same for all subjects) or a column name in df
        y0 = y0 if isinstance(y0, float) else subject_data[y0].iloc[0]
        if pd.isna(y0) or np.isnan(y0):
            print(f"Subject {subject} has NaN y0, skipping.")
            continue
        
        #vary0 may be a float (same for all subjects) or a column name in df or None
        if vary0 is not None:
            vary0 = vary0 if isinstance(vary0, float) else subject_data[vary0].iloc[0]
            if pd.isna(vary0) or np.isnan(vary0):
                print(f"Subject {subject} has NaN vary0, skipping.")
                continue
        
        # plot MBP data
        ax.plot(
            subject_data[x],
            subject_data[y],
            color="#4A4A4A",
            alpha=0.8,
            label=y
        )
        
        fitted_data = pd.DataFrame({
                            x: np.arange(xmin,xmax, 0.01),
                            y: np.nan
                            })

        fitted_data = create_fitted_data_same(fitted_data, x, 
                                              (xmin, xmax), 
                                              y,
                                              model, *pfit)
        
        ax.plot(
            fitted_data[x],
            fitted_data[y],
            color='red',
            linestyle='--',
            linewidth=1.5,
            label='Fit'
            )

        pstr = ' '.join([f"{k}={v:.1g}" 
                         for k,v in zip(names, pfit, strict=True)]
                        )
        ax.annotate(
            pstr,
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=7,
            ha='left',
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8)
        )
        
        finalize_mbp_axes(ax,hline=y0,
                          hband=np.sqrt(vary0) if vary0 is not None else None
                          )
        ax.set_title(f"{subject} - {condition}")
    
    for ax in axes:
        # Consistency: vertical line + stim bar + labels
        add_stim_to_ax(ax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ax.get_ylim()[1])
        #ax.legend()
        
    plt.tight_layout()
    return fig
        
        
def plot_mbpfit_parameters_group(pfit_df, index2group, param_interpretation={}):
    # Add group column
    grpcol = pfit_df.copy()
    grpcol.reset_index(names='subj', inplace=True)
    grpcol['Group'] = grpcol['subj'].map(index2group)
    grpcol.set_index('subj', inplace=True)
    _pfit_df = pfit_df.copy()
    _pfit_df['Group'] = grpcol['Group']

    # plot, swarm
    fig,axs = plt.subplots(nrows=(_pfit_df.shape[1]-1+1)//2,ncols=2,figsize=(12,12))
    axs = axs.T.flatten()
    for i, (ax, col) in enumerate(zip(axs, _pfit_df.columns[:-1], strict=False)):
        sns.swarmplot(x="Group", y=col, hue="Group", data=_pfit_df, 
                      dodge=True, ax=ax, size=3,
                      legend=(i==len(_pfit_df.columns)-2)
)
        ax.set_title(f"{col} {param_interpretation.get(col, '')}")
    
    for ax in axs[_pfit_df.shape[1]-1:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show(block=False)
    
    return fig

# plot all the fittings together — single figure: first row = averaged, then one row per group (separated)
def plot_mbpfit_fitted_group(pfit_df, subj_to_group, condition, model, names, y, add_stim=True,
                             xmin = TMIN, xmax = TMAX, dx=0.01,axes=None):
    # build fitted curves per subject
    all_fits = []
    x = np.arange(xmin, xmax, dx)
    for subject in np.unique(pfit_df.index):
        group = subj_to_group.get(subject, 'Other')
        pfit = pfit_df.loc[subject, names].to_numpy()
        base_df = pd.DataFrame({
            "Subject": [subject] * len(x),
            "Group": [group] * len(x),
            "x": x,
            y: np.nan
        })
        fitted = create_fitted_data_same(base_df, 'x', (xmin,xmax), y, model, *pfit)
        all_fits.append(fitted)
    all_fits = pd.concat(all_fits, ignore_index=True)

    # determine groups to plot (preserve common order if possible)
    preferred_order = ['HC', 'eMCS', 'pDoC']
    # ensure groups_present reflects only groups in data and keep preferred order first if present
    present = [g for g in preferred_order if g in all_fits['Group'].unique()] + \
              [g for g in sorted(all_fits['Group'].unique()) if g not in preferred_order]
    groups = present

    # color map (fallbacks if constants not defined)
    cmap = {
        'HC': globals().get('C_HC', '#0072B2'),
        'eMCS': globals().get('C_EMCS', '#009E73'),
        'pDoC': globals().get('C_PDOC', '#D55E00'),
    }

    n_axes = 1 + len(groups)
    if axes is None:
        fig, axes = plt.subplots(n_axes, 1, figsize=(10, 3 * n_axes), sharex=True, sharey=True)
        if n_axes == 1:
            axes = [axes]
    # Allow passing a larger preallocated grid; use the first needed rows.
    if hasattr(axes, "shape"):
        if axes.shape[0] < n_axes:
            raise AssertionError(f"First dim of axes is {axes.shape[0]}, expected at least {n_axes}")
        if axes.shape[0] > n_axes:
            axes = axes[:n_axes]

    # Averaged plot (top)
    ax_avg = axes[0]
    sns.lineplot(
        data=all_fits,
        x='x',
        y=y,
        hue='Group',
        hue_order=groups,
        estimator='mean',
        errorbar=('ci', 95),
        palette=[cmap.get(g, '#888888') for g in groups],
        ax=ax_avg,
        alpha=0.8,
        legend=False,
    )
    ax_avg.set_title(f'Average fitted {y} curves per Group — model: {condition}')
    finalize_mbp_axes(ax_avg)
    ax_avg.set_xlabel('Time from stimulus onset (ms)')
    ax_avg.set_xlim(xmin, xmax)
    #ax_avg.legend(title='Group', loc='upper right')

    # One axis per group: individual subject fits for that group
    for i, group in enumerate(groups):
        ax = axes[i + 1]
        dfg = all_fits[all_fits['Group'] == group]
        if dfg.empty:
            ax.set_title(f'Group {group} — no data')
            ax.set_xlim(xmin, xmax)
            continue

        # plot individual subject fits (no aggregation)
        sns.lineplot(
            data=dfg,
            x='x',
            y=y,
            units='Subject',
            estimator=None,
            ax=ax,
            color=cmap.get(group, '#888888'),
            alpha=0.6,
        )

        ax.set_title(f'{group} (n={dfg["Subject"].nunique()})')
        finalize_mbp_axes(ax)
        ax.set_xlim(xmin, xmax)
        #ax.legend()

    if add_stim:
        for ax in axes:
            add_stim_to_ax(ax)
    
    return axes

# %% [markdown]
# ## Data
# 
# #### Note: skip and go to loading and below to avoid computation

# %% [markdown]
# #### Raw stimuli and blink data

# %%
#  Get data paths

res_folder = os.getenv('MBP_INPUT_RESULTS_DIR', './results')
data_folder = './data'
condition = "ODDBALL" # Condition for oddball trials

def load_subject_metadata(res_folder, data_folder):
    """Load legacy enrolment metadata or public-dataset demographics."""
    enrolled_csv = os.path.join(res_folder, 'enrolled_subjects.csv')
    demo_csv = os.path.join(data_folder, 'demographics.csv')

    if os.path.exists(enrolled_csv):
        return pd.read_csv(enrolled_csv)
    if not os.path.exists(demo_csv):
        raise FileNotFoundError(
            "Subject metadata not found. Expected either "
            f"'{enrolled_csv}' or '{demo_csv}'."
        )

    df_demo = pd.read_csv(demo_csv).copy()
    if not {'Subject', 'Group'}.issubset(df_demo.columns):
        raise ValueError(f"'{demo_csv}' must contain Subject and Group columns.")

    # Public demographics use numeric subject IDs, while result files use sub-00.
    def _to_result_subject_id(subject):
        try:
            return f"sub-{int(float(subject)):02d}"
        except (TypeError, ValueError):
            return str(subject).strip()

    df_demo['Subject'] = df_demo['Subject'].map(_to_result_subject_id)
    return df_demo


def build_subject_group_map(res_folder, data_folder):
    df_subjects = load_subject_metadata(res_folder, data_folder)
    return df_subjects.set_index('Subject')['Group'].to_dict()

subj_to_group = build_subject_group_map(res_folder, data_folder)

# blinks fnames 
blinks_files = glob(f"**/*_{condition}_blinks.csv",
                    root_dir=res_folder, recursive=True)
# Support BIDS-like naming produced by eog_analysis.py, e.g.:
# sub-00_task-ODDBALL_eog_blinks.csv
blinks_files.extend(
    glob(f"**/*task-{condition}*blinks.csv",
         root_dir=res_folder, recursive=True)
)
# events fnames 
events_files = glob(f"**/*_{condition}_stims.csv",
                    root_dir=res_folder, recursive=True)
events_files.extend(
    glob(f"**/*task-{condition}*stims.csv",
         root_dir=res_folder, recursive=True)
)
# resting blinks fnames 
rsblinks_files = glob(f"**/*_RESTING_blinks.csv",
                     root_dir=res_folder, recursive=True)
rsblinks_files.extend(
    glob("**/*task-RESTING*blinks.csv",
         root_dir=res_folder, recursive=True)
)

# de-duplicate while preserving order
blinks_files = list(dict.fromkeys(blinks_files))
events_files = list(dict.fromkeys(events_files))
rsblinks_files = list(dict.fromkeys(rsblinks_files))


# create base-blinks_file-events_file-resting_blinks_file association as a dictionary
# keyed by base name ([group]_[subj]) and value is the three files
def _extract_base(fname: str, condition: str):
    stem = os.path.splitext(os.path.basename(fname))[0]
    # Legacy: <base>_ODDBALL_blinks / <base>_ODDBALL_stims
    legacy_tok = f"_{condition}"
    if legacy_tok in stem:
        return stem.split(legacy_tok)[0]
    # BIDS-like: <base>_task-ODDBALL_eog_blinks / ..._stims
    bids_tok = f"_task-{condition}"
    if bids_tok in stem:
        return stem.split(bids_tok)[0]
    # RESTING variants
    if "_RESTING" in stem:
        return stem.split("_RESTING")[0]
    if "_task-RESTING" in stem:
        return stem.split("_task-RESTING")[0]
    return stem

bases = [_extract_base(f, condition) for f in blinks_files]
bases.extend([_extract_base(f, condition) for f in events_files])
bases.extend([_extract_base(f, condition) for f in rsblinks_files])
bases = sorted(set(bases)) 

blinks_events_dict = {base: (None, None, None, None) for base in bases}
for f in blinks_files:
    base = _extract_base(f, condition)
    blinks_events_dict[base] = (
        f,  # blinks file
        blinks_events_dict[base][1],  # events file
        blinks_events_dict[base][2]  # signals file
    )
for f in events_files:
    base = _extract_base(f, condition)
    blinks_events_dict[base] = (
        blinks_events_dict[base][0],  # blinks file
        f,  # events file
        blinks_events_dict[base][2]   # signals file
    )
for f in rsblinks_files:
    base = _extract_base(f, condition)
    blinks_events_dict[base] = (
        blinks_events_dict[base][0],  # blinks file
        blinks_events_dict[base][1],  # events file
        f   # resting blinks file
    )
    
print("Data to be processed:")
allpaths = pd.DataFrame.from_dict(blinks_events_dict)
allpaths['Kind'] = ['OD BLK', 'OD STIM', 'RS BLK']
allpaths.set_index('Kind', inplace=True)
allpaths = allpaths.T
display(allpaths)

# Diagnostic: verify subject IDs found in files can be mapped to enrolled groups
if not allpaths.empty:
    matched_groups = pd.Series([subj_to_group.get(s, 'Other') for s in allpaths.index], name='Group')
    print("Detected subjects by mapped group:")
    print(matched_groups.value_counts(dropna=False))
    if (matched_groups == 'Other').all():
        print("WARNING: all detected subjects map to 'Other'. Subject IDs in filenames do not match enrolled_subjects.csv.")

if allpaths.empty:
    raise RuntimeError(
        f"No input blink/stim files found under '{res_folder}'. "
        f"Matched counts -> OD blinks: {len(blinks_files)}, OD stims: {len(events_files)}, "
        f"RESTING blinks: {len(rsblinks_files)}."
    )



# %% [markdown]
# #### Rayleigh phase-locking analysis (raw blink timings, no MBP binning)

# %%
rayleigh_rows = []
rayleigh_lags_long = []

for base, row in tqdm(allpaths.iterrows(), desc="Rayleigh phase-locking"):
    grp = subj_to_group.get(base, 'Other')
    events_file = os.path.join(res_folder, row['OD STIM'])

    try:
        events_df = pd.read_csv(events_file)
        stimuli = _read_col_anycase(events_df, 'onset').to_numpy(dtype=float)
    except Exception as e:
        print(f"[Rayleigh] Cannot read stimuli for {base}: {e}")
        continue

    for cond_name, blink_col in [('ODDBALL', 'OD BLK'), ('RESTING', 'RS BLK')]:
        blinks_file = os.path.join(res_folder, row[blink_col])
        try:
            blinks_df = pd.read_csv(blinks_file)
            blinks = _read_col_anycase(blinks_df, 'Peak').to_numpy(dtype=float)
        except Exception as e:
            print(f"[Rayleigh] Cannot read blinks for {base} {cond_name}: {e}")
            rayleigh_rows.append({
                'Group': grp,
                'Condition': cond_name,
                'Subject': base,
                'N_stimuli': int(stimuli.size),
                'N_blinks_window': 0,
                'Rbar': np.nan,
                'Z': np.nan,
                'P': np.nan,
                'Preferred latency (s)': np.nan,
            })
            continue

        rel_t = collect_relative_blinks(stimuli, blinks, tmin=TMIN, tmax=TMAX)
        wrapped_t = wrap_to_window(rel_t, tmin=TMIN, tmax=TMAX)
        phases = _to_phase(wrapped_t, tmin=TMIN, tmax=TMAX)
        rt = rayleigh_test(phases)

        if phases.size > 0:
            mu = np.angle(np.mean(np.exp(1j * phases)))
            mu = np.mod(mu, 2 * np.pi)
            pref_t = TMIN + (mu / (2 * np.pi)) * (TMAX - TMIN)
        else:
            pref_t = np.nan

        rayleigh_rows.append({
            'Group': grp,
            'Condition': cond_name,
            'Subject': base,
            'N_stimuli': int(stimuli.size),
            'N_blinks_window': int(rt['n']),
            'Rbar': rt['rbar'],
            'Z': rt['z'],
            'P': rt['p'],
            'Preferred latency (s)': pref_t,
        })

        if wrapped_t.size > 0:
            rayleigh_lags_long.append(pd.DataFrame({
                'Group': grp,
                'Condition': cond_name,
                'Subject': base,
                'RelTimeWrapped': wrapped_t
            }))

df_rayleigh = pd.DataFrame(rayleigh_rows)

def _bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if not np.any(valid):
        return q
    pv = pvals[valid]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_valid = np.empty_like(pv)
    q_valid[order] = q_ranked
    q[valid] = q_valid
    return q

# FDR-BH correction within each Group (across all subject-level Rayleigh p-values in that group)
df_rayleigh['P_fdr_bh_group'] = np.nan
for grp, idx in df_rayleigh.groupby('Group', dropna=False).groups.items():
    pvals = df_rayleigh.loc[idx, 'P'].to_numpy(dtype=float)
    df_rayleigh.loc[idx, 'P_fdr_bh_group'] = _bh_fdr(pvals)

display(df_rayleigh)
save_table_csv(df_rayleigh, 'rayleigh_subject_level.csv')
df_rayleigh.to_csv(os.path.join(CACHE_DIR, 'rayleigh_subject_level.csv'), index=False)

if rayleigh_lags_long:
    df_rayleigh_lags = pd.concat(rayleigh_lags_long, ignore_index=True)
else:
    df_rayleigh_lags = pd.DataFrame(columns=['Group', 'Condition', 'Subject', 'RelTimeWrapped'])
df_rayleigh_lags.to_csv(os.path.join(CACHE_DIR, 'rayleigh_relative_lags_long.csv'), index=False)

rayleigh_summ_rows = []
for (grp, cond), d in df_rayleigh.groupby(['Group', 'Condition'], dropna=False):
    p_unc = d['P'].to_numpy(dtype=float)
    p_cor = d['P_fdr_bh_group'].to_numpy(dtype=float)
    p_unc = p_unc[np.isfinite(p_unc)]
    p_cor = p_cor[np.isfinite(p_cor)]
    n_subj = int(d['Subject'].nunique())

    n_sig_unc = int(np.sum(p_unc < ALPHA))
    n_sig_cor = int(np.sum(p_cor < ALPHA))

    # Fisher composite tests on subject-level p-values (uncorrected and BH-corrected within-group).
    if p_unc.size > 0:
        fisher_stat_unc, fisher_p_unc = stats.combine_pvalues(p_unc, method='fisher')
        fisher_sig_unc = bool(fisher_p_unc < ALPHA)
    else:
        fisher_stat_unc, fisher_p_unc, fisher_sig_unc = np.nan, np.nan, False
    if p_cor.size > 0:
        fisher_stat_cor, fisher_p_cor = stats.combine_pvalues(p_cor, method='fisher')
        fisher_sig_cor = bool(fisher_p_cor < ALPHA)
    else:
        fisher_stat_cor, fisher_p_cor, fisher_sig_cor = np.nan, np.nan, False

    # Circular mean/std for preferred latency.
    t_pref = d['Preferred latency (s)'].to_numpy(dtype=float)
    t_pref = t_pref[np.isfinite(t_pref)]
    if t_pref.size > 0:
        t_pref_mean = float(stats.circmean(t_pref, low=TMIN, high=TMAX))
        # scipy circstd is circular by construction; keep seconds unit by using same low/high.
        t_pref_std = float(stats.circstd(t_pref, low=TMIN, high=TMAX))
    else:
        t_pref_mean, t_pref_std = np.nan, np.nan

    # Binomial responder tests (uncorrected and BH-corrected within-group).
    binom_p_unc = float(stats.binom.sf(n_sig_unc - 1, n_subj, ALPHA)) if n_subj > 0 else np.nan
    binom_p_cor = float(stats.binom.sf(n_sig_cor - 1, n_subj, ALPHA)) if n_subj > 0 else np.nan

    rayleigh_summ_rows.append({
        'Group': grp,
        'Condition': cond,
        'N_subjects': n_subj,
        'N_sig_uncorrected': n_sig_unc,
        'N_sig_fdr_bh_within_group': n_sig_cor,
        'P < 0.05 uncorr (nsig/ntot)': f'{n_sig_unc}/{n_subj}',
        'P < 0.05 fdr-bh-within-group (nsig/ntot)': f'{n_sig_cor}/{n_subj}',
        'Median_P_uncorrected': float(np.nanmedian(p_unc)) if p_unc.size > 0 else np.nan,
        'Mean_Rbar': float(np.nanmean(d['Rbar'].to_numpy(dtype=float))) if len(d) > 0 else np.nan,
        'Std_Rbar': float(np.nanstd(d['Rbar'].to_numpy(dtype=float), ddof=1)) if len(d) > 1 else 0.0,
        'Mean_t_pref_s': t_pref_mean,
        'Std_t_pref_s': t_pref_std,
        'Fisher_X2_uncorrected': float(fisher_stat_unc) if np.isfinite(fisher_stat_unc) else np.nan,
        'Fisher_p_uncorrected': float(fisher_p_unc) if np.isfinite(fisher_p_unc) else np.nan,
        'Fisher_significant_uncorrected': bool(fisher_sig_unc),
        'Fisher_X2_fdr_bh_within_group': float(fisher_stat_cor) if np.isfinite(fisher_stat_cor) else np.nan,
        'Fisher_p_fdr_bh_within_group': float(fisher_p_cor) if np.isfinite(fisher_p_cor) else np.nan,
        'Fisher_significant_fdr_bh_within_group': bool(fisher_sig_cor),
        'Binom_tail_p_uncorrected': float(binom_p_unc) if np.isfinite(binom_p_unc) else np.nan,
        'Binom_significant_uncorrected': bool(np.isfinite(binom_p_unc) and (binom_p_unc < ALPHA)),
        'Binom_tail_p_fdr_bh_within_group': float(binom_p_cor) if np.isfinite(binom_p_cor) else np.nan,
        'Binom_significant_fdr_bh_within_group': bool(np.isfinite(binom_p_cor) and (binom_p_cor < ALPHA)),
    })

rayleigh_summary = pd.DataFrame(rayleigh_summ_rows)

def _fmt_mean_std(mu, sd, fmt='%.3f'):
    if pd.isna(mu):
        return np.nan
    if pd.isna(sd):
        sd = 0.0
    return f"{fmt % mu} +/- {fmt % sd}"

rayleigh_summary['Rbar (mean +/- std)'] = rayleigh_summary.apply(
    lambda r: _fmt_mean_std(r['Mean_Rbar'], r['Std_Rbar'], fmt='%.3f'), axis=1
)
rayleigh_summary['t_pref (s, mean +/- std)'] = rayleigh_summary.apply(
    lambda r: _fmt_mean_std(r['Mean_t_pref_s'], r['Std_t_pref_s'], fmt='%.3f'), axis=1
)

rayleigh_summary = rayleigh_summary[
    [
        'Group',
        'Condition',
        'P < 0.05 uncorr (nsig/ntot)',
        'P < 0.05 fdr-bh-within-group (nsig/ntot)',
        'Median_P_uncorrected',
        'Rbar (mean +/- std)',
        't_pref (s, mean +/- std)',
        'Fisher_X2_uncorrected',
        'Fisher_p_uncorrected',
        'Fisher_significant_uncorrected',
        'Fisher_X2_fdr_bh_within_group',
        'Fisher_p_fdr_bh_within_group',
        'Fisher_significant_fdr_bh_within_group',
        'Binom_tail_p_uncorrected',
        'Binom_significant_uncorrected',
        'Binom_tail_p_fdr_bh_within_group',
        'Binom_significant_fdr_bh_within_group',
    ]
]
display(rayleigh_summary)
save_table_csv(rayleigh_summary, 'rayleigh_group_summary.csv')

ray_groups_present = [g for g in ['HC', 'eMCS', 'pDoC'] if g in df_rayleigh_lags['Group'].unique()]
if len(ray_groups_present) == 0:
    ray_groups_present = sorted(df_rayleigh['Group'].dropna().unique())

if len(ray_groups_present) > 0 and not df_rayleigh.empty:
    conds = ['ODDBALL'] + (['RESTING'] if RAYLEIGH_PLOT_RESTING else [])
    nrows = len(ray_groups_present)
    ncols = len(conds)
    fig_w = 5.31
    fig_h = fig_w/3*2
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axs = np.array(axs, dtype=object)
    if axs.ndim == 1:
        if nrows == 1:
            axs = axs.reshape(1, ncols)
        else:
            axs = axs.reshape(nrows, 1)

    g2c = {'HC': C_HC, 'eMCS': C_EMCS, 'pDoC': C_PDOC}
    for g in ray_groups_present:
        g2c.setdefault(g, '#4A4A4A')

    for i, grp in enumerate(ray_groups_present):
        for j, cond_name in enumerate(conds):
            ax = axs[i, j]
            d = df_rayleigh[
                (df_rayleigh['Group'] == grp)
                & (df_rayleigh['Condition'] == cond_name)
            ].copy()
            if not d.empty:
                d = d.dropna(subset=['Preferred latency (s)', 'Rbar', 'P'])
                if not d.empty:
                    x = d['Preferred latency (s)'].to_numpy(dtype=float)
                    y = d['Rbar'].to_numpy(dtype=float)
                    p = d['P_fdr_bh_group'].to_numpy(dtype=float)
                    sig_mask = np.isfinite(p) & (p < ALPHA)
                    # Non-significant subjects
                    nonsig_alpha = 1
                    if np.any(~sig_mask):
                        ax.vlines(x[~sig_mask], 0.0, y[~sig_mask], colors=g2c.get(grp, '#4A4A4A'),
                                  linewidth=1.2, alpha=nonsig_alpha)
                        ax.scatter(x[~sig_mask], y[~sig_mask], color=g2c.get(grp, '#4A4A4A'),
                                   s=12, zorder=3, alpha=nonsig_alpha, marker='d')
                    # Significant subjects
                    if np.any(sig_mask):
                        ax.vlines(x[sig_mask], 0.0, y[sig_mask], colors=g2c.get(grp, '#4A4A4A'),
                                  linewidth=1.2, alpha=1.0)
                        ax.scatter(x[sig_mask], y[sig_mask], color=g2c.get(grp, '#4A4A4A'),
                                   s=12, zorder=3, alpha=1.0, marker='d')

                    # Mark significant subjects with rotated asterisk above the bar tip.
                    ypad = 0
                    for xi, yi, pi in zip(x, y, p, strict=True):
                        if np.isfinite(pi) and (pi < ALPHA):
                            ax.annotate(
                                '*',
                                xy=(xi, yi + ypad),
                                ha='center',
                                va='bottom',
                                fontsize=10,
                                color='k',
                            )
            ax.set_xlim(-0.52, 0.52)
            ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
            ax.set_ylim(0, 0.4)
            if j == 0:
                ax.set_ylabel(r'$\bar{R}$'+f' - {grp}')
            #ax.legend([],[],title=cond_name)

    # Add stimulus annotation at the end so bars/icons share the same y placement.
    for ax in axs.flatten():
        add_stim_to_ax(ax)

    # Keep x-label only on the last row.
    for i in range(max(nrows - 1, 0)):
        for j in range(ncols):
            axs[i, j].set_xlabel('')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'rayleigh_wrapped_lag_density.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'rayleigh_wrapped_lag_density.eps'), bbox_inches='tight')
    plt.show(block=False)

# %% [markdown]
# #### Counts dataframe with no overlap for counts testing (includes Resting for negative example testing!)

# %%
# creation of df_counts
df_counts = []

for base, row in tqdm(allpaths.iterrows(), desc="Computing relative blink counts"):
    # ========= Load the data from CSV files ===========================
    blinks_file = os.path.join(res_folder, row['OD BLK'])
    events_file = os.path.join(res_folder, row['OD STIM'])
    rsblinks_file = os.path.join(res_folder, row['RS BLK'])
    
    grp = subj_to_group.get(base, 'Other')
    
    def _counts_helper(blinks_csv, events_csv, base, grp, condition):
        try:
            blinks_df = pd.read_csv(blinks_csv)
            blinks=blinks_df['Peak'].to_numpy()
            
            if blinks.size == 0:
                raise ValueError("Blinks data is empty.")
            
            events_df = pd.read_csv(events_csv)
            stimuli=events_df['onset'].to_numpy()
            
            times, counts, n = compute_counts(stimuli,blinks, 
                                            tmin=TMIN, tmax=TMAX,
                                            overlap_ratio=0, # no overlap for χ²!
                                            nbin=NBIN)
            
            return pd.DataFrame({'Group': grp,
                                'Condition': condition,
                                'Subject': base,
                                'Time':times,
                                'Counts':counts,
                                'N_blinks': n
                                })

        except Exception as e:
            grp = subj_to_group.get(base, 'Other')
            
            print("Error while loading data", e)
            bin_width = (TMAX - TMIN) / NBIN
            edges = np.arange(TMIN, TMAX + bin_width, bin_width)
            times = edges[:-1] + bin_width / 2.0

            return pd.DataFrame({'Group': grp,
                                        'Condition': condition,
                                        'Subject': base,
                                        'Time':times,
                                        'Counts':np.nan,
                                        'N_blinks': np.nan
                                        })
    
    # Compute counts on oddball 
    df_counts.append(_counts_helper(blinks_file, events_file, base, grp, 'ODDBALL'))

    # Compute counts on resting-state
    df_counts.append(_counts_helper(rsblinks_file, events_file, base, grp, 'RESTING'))
        

df_counts = pd.concat(df_counts)
display(df_counts)
df_counts.to_csv(os.path.join(CACHE_DIR, 'df_counts.csv'), index=False)


# %% [markdown]
# #### MBP dataframe with overlap for fitting 

# %%
# creation of df_mbp
df_mbp = []

for base, row in tqdm(allpaths.iterrows(), desc="Computing MBP"):
    # ========= Load the data from CSV files ===========================
    blinks_file = os.path.join(res_folder, row['OD BLK'])
    events_file = os.path.join(res_folder, row['OD STIM'])
    rsblinks_file = os.path.join(res_folder, row['RS BLK'])
    
    grp = subj_to_group.get(base, 'Other')
    
    def _mbp_helper(blinks_csv, events_csv, base, grp, condition):
        try:
            blinks_df = pd.read_csv(blinks_csv)
            blinks=blinks_df['Peak'].to_numpy()
            
            if blinks.size == 0:
                raise ValueError("Blinks data is empty.")
            
            events_df = pd.read_csv(events_csv)
            stimuli=events_df['onset'].to_numpy()
            
            times, mbp, n = compute_mbp(stimuli,blinks, 
                                            tmin=TMIN, tmax=TMAX,
                                            overlap_ratio=OVBIN, # overlap for MBP smoothing
                                            nbin=NBIN)
            
            return pd.DataFrame({'Group': grp,
                                'Condition': condition,
                                'Subject': base,
                                'Time':times,
                                'MBP':mbp,
                                'E(MBP_0)': null_mbp(B=NBIN),
                                'Var(MBP_0)': null_mbp_var(n, B=NBIN),
                                'N_blinks': n
                                })

        except Exception as e:
            grp = subj_to_group.get(base, 'Other')
            
            print("Error while loading data", e)
            bin_width = (TMAX - TMIN) / NBIN
            edges = np.arange(TMIN, TMAX + bin_width, bin_width)
            times = edges[:-1] + bin_width / 2.0

            return pd.DataFrame({'Group': grp,
                                        'Condition': condition,
                                        'Subject': base,
                                        'Time':times,
                                        'Counts':np.nan,
                                        'E(MBP_0)': np.nan,
                                        'Var(MBP_0)': np.nan,
                                        'N_blinks': np.nan
                                        })
    
    # Compute counts on oddball 
    df_mbp.append(_mbp_helper(blinks_file, events_file, base, grp, 'ODDBALL'))

    # Compute counts on resting-state
    df_mbp.append(_mbp_helper(rsblinks_file, events_file, base, grp, 'RESTING'))
        

df_mbp = pd.concat(df_mbp)
display(df_mbp)
df_mbp.to_csv(os.path.join(CACHE_DIR, 'df_mbp.csv'), index=False)

# %% [markdown]
# ## Re-loading for hot start

# %%
df_mbp = pd.read_csv(os.path.join(CACHE_DIR, 'df_mbp.csv'))
df_counts = pd.read_csv(os.path.join(CACHE_DIR, 'df_counts.csv'))
condition = "ODDBALL" # Condition for oddball trials
res_folder = os.getenv('MBP_INPUT_RESULTS_DIR', './results')
data_folder = './data'
subj_to_group = build_subject_group_map(res_folder, data_folder)

# %%
# visualize MBP raw data with line plot, units = Subjects, estimator = None
plt.close('all')
present_groups = list(pd.Series(df_mbp['Group'].dropna().unique()))
preferred_groups = ['HC', 'eMCS', 'pDoC']
plot_groups = [g for g in preferred_groups if g in present_groups]
if len(plot_groups) == 0:
    plot_groups = sorted(present_groups)

fig, axs = plt.subplots(len(plot_groups), 2, figsize=(5, 2 * max(1, len(plot_groups))), sharex=True, sharey=True)
if len(plot_groups) == 1:
    axs = np.array([axs])

palette_map = {'HC': C_HC, 'eMCS': C_EMCS, 'pDoC': C_PDOC}
for g in plot_groups:
    palette_map.setdefault(g, '#4A4A4A')

for j,cond in enumerate(['RESTING','ODDBALL']):
    for i, g in enumerate(plot_groups):
        ax = axs[i,j]
        sns.lineplot(data=df_mbp.query(f'Condition == "{cond}"').query(f'Group == "{g}"'), 
                    x='Time', y='MBP', hue='Group', style='Group',
                    palette=palette_map,
                    markers='.', err_kws={'alpha':0.3},
                    estimator='median', errorbar=('ci', 95),
                    ax=ax, legend=False)
        finalize_mbp_axes(ax)
        ax.set_xlim(TMIN, TMAX)
        ax.set_title(f'{g} - {cond}')
        
for ax in axs.flatten():
    add_stim_to_ax(ax)
handles, labels = axs[0,0].get_legend_handles_labels()
if len(handles) > 0:
    plt.legend(handles=handles, labels=labels, title='Group', loc='upper right')
plt.tight_layout()
plt.show(block=False)


# %% [markdown]
# ## Data-driven analysis of binned blink counts (Huber 2022)

# %%
# Summarize χ² results by Group × Condition, separately for Pre and Post: KS, binomial-tail, PP-plot

# --- 1) Build subject_window_df with Condition included ---
counts = df_counts.copy()
counts['Counts'] = pd.to_numeric(counts['Counts'], errors='coerce').fillna(0)
counts['Time']   = pd.to_numeric(counts['Time'],   errors='coerce')
counts = counts.dropna(subset=['Subject','Time','Counts'])
counts = counts[counts['Time'] != 0]
counts['Window'] = np.where(counts['Time'] < 0, 'Pre', 'Post')
# keep only expected conditions (adjust if you have others)
if 'Condition' in counts.columns:
    counts = counts[counts['Condition'].isin(['ODDBALL','RESTING'])].copy()
else:
    raise KeyError("df_counts must include a 'Condition' column with values like 'ODDBALL' and 'RESTING'.")

def chi2_uniform_from_counts(bin_counts: np.ndarray):
    k = bin_counts.size
    n = int(bin_counts.sum())
    if k <= 1 or n == 0:
        return np.nan, k-1, np.nan, np.nan, False
    expected = n / k
    stat = ((bin_counts - expected) ** 2 / np.where(expected > 0, expected, np.nan)).sum()
    df = k - 1
    p  = 1.0 - stats.chi2.cdf(stat, df)
    return float(stat), int(df), float(p), float(expected), bool(expected >= 5)

rows = []
for (grp, sid, cond, win), df_sw in counts.groupby(['Group','Subject','Condition','Window']):
    bin_counts = df_sw.sort_values('Time')['Counts'].to_numpy(float)
    chi2, dfree, pval, expected, adequacy = chi2_uniform_from_counts(bin_counts)
    rows.append({
        'group': grp, 'subject': sid, 'condition': cond, 'window': win,
        'n_blinks': int(bin_counts.sum()), 'n_bins': int(bin_counts.size),
        'expected_per_bin': expected, 'chi2': chi2, 'df': dfree, 'p': pval, 'adequacy_flag': adequacy
    })
subject_window_df = pd.DataFrame(rows)

# --- 1.5) PP plotting


from matplotlib.lines import Line2D
def pp_data(pvals):
    
    n_grid = len(pvals)
    p = np.asarray(pvals, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.array([]), np.array([])
    p.sort()
    t = np.linspace(0, 1, n_grid)                 # thresholds
    x = t                                         # theoretical CDF under Uniform
    y = np.searchsorted(p, t, side='right') / p.size  # empirical CDF at t
    return x, y

# ---------- plotting ----------
_group_colors = {
    'HC':  globals().get('C_HC',  None),
    'eMCS':globals().get('C_EMCS',None),
    'pDoC':globals().get('C_PDOC',None),
}
default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2'])
for i, g in enumerate(['HC','eMCS','pDoC']):
    if _group_colors.get(g) is None:
        _group_colors[g] = default_cycle[i % len(default_cycle)]

# marker map by (Condition, Window)
MARKER = {
    ('ODDBALL','Pre') : 's',   # square (filled in legend)
    ('ODDBALL','Post'): 'D',   # diamond (filled in legend)
    ('RESTING','Pre') : 'x',   # x (unfilled)
    ('RESTING','Post'): '+',   # plus (unfilled)
}

fig, axes = plt.subplots(3, 2, figsize=(5.31, 7.5), sharex=False, sharey=False)
groups = ['HC','eMCS','pDoC']
wins = ['Pre','Post']

for i, grp in enumerate(groups):
    for j, win in enumerate(wins):
        ax = axes[i, j]
        ax.plot([0,1],[0,1], linestyle='--', color='0.6', linewidth=0.8)
        ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
        ax.set_title(grp+' - '+win, fontsize=10, pad=2)
        # only label axes on left column / bottom row for cleanliness
        if i == 2:
            ax.set_xlabel('Theoretical CDF (Uniform)', fontsize=9, labelpad=2)
        if j == 0:
            ax.set_ylabel('Empirical CDF', fontsize=9, labelpad=2)
        ax.tick_params(axis='both', which='both', labelsize=9, length=3)

        for cond in ['ODDBALL','RESTING']:
            sel = subject_window_df[(subject_window_df['group']==grp) &
                                    (subject_window_df['condition']==cond) & 
                                    (subject_window_df['window']==win)]
            pvals = sel['p'].to_numpy()
            x,y = pp_data(pvals)

            color  = _group_colors[grp]
            marker = MARKER[(cond, win)]
            h, = ax.plot(x, y, linestyle='-', linewidth=0.5, marker=marker,
                         markerfacecolor=[0,0,0,0], # keep main plot markers unfilled as before
                         markeredgecolor=color, 
                         color=color,
                         markersize=4.5)
        # add small black marker legend in the top-row plots
        if i == 0:
            legend_handles = []
            legend_labels = []
            # consider which markers should be shown for this column (win)
            for cond in ['ODDBALL','RESTING']:
                m = MARKER[(cond, win)]
                # markers that visually support a filled face:
                # filledable = set(['s','D','o','^','v','<','>','P','X'])
                #mface = 'black' if m in filledable else 'none'
                handle = Line2D([0],[0], marker=m, linestyle='', color='black',
                                markerfacecolor='none', markeredgecolor='black', markersize=6)
                legend_handles.append(handle)
                legend_labels.append(cond.capitalize())
            ax.legend(legend_handles, legend_labels, loc='upper left', frameon=False, fontsize=8)

fig.savefig(os.path.join(FIGURES_DIR, 'chi2_pp_plots.svg'), bbox_inches='tight')
fig.savefig(os.path.join(FIGURES_DIR, 'chi2_pp_plots.eps'), bbox_inches='tight')
plt.show(block=False)

# --- 2) Group-level aggregation: KS on p-values + binomial tail ---
def ks_uniform(pvals: np.ndarray):
    pvals = np.asarray(pvals)
    pvals = pvals[np.isfinite(pvals)]
    if pvals.size == 0:
        return np.nan, np.nan
    D, p = stats.kstest(pvals, 'uniform')
    return float(D), float(p)

def binomial_tail(m: int, n: int, alpha: float = ALPHA):
    if n <= 0:
        return np.nan
    return float(stats.binom.sf(m - 1, n, alpha))  # P[X >= m]

summ_rows = []
for (grp, cond, win), df_gcw in subject_window_df.groupby(['group','condition','window']):
    ps = df_gcw['p'].to_numpy()
    ps = ps[np.isfinite(ps)]
    n = int(ps.size)
    m = int((ps < ALPHA).sum())
    D, pks = ks_uniform(ps)
    pbin = binomial_tail(m, n, ALPHA)
    summ_rows.append({
        'group': grp, 'condition': cond, 'window': win,
        'n_subjects': n,
        f'n_sig_p<{ALPHA}': m,
        'frac_sig': (m / n) if n > 0 else np.nan,
        'KS_D': D,
        'KS_p': pks,
        'binom_tail_p': pbin
    })

group_summary_df = (pd.DataFrame(summ_rows)
                    .sort_values(['group','condition','window'])
                    .reset_index(drop=True))
_cond_map = {'ODDBALL': 'OD', 'RESTING': 'RS'}

def make_compact_table(df, window):
    dfw = df[df['window'] == window].copy()
    dfw['Cond'] = dfw['condition'].map(_cond_map)
    dfw['p_X2 < 0.05 (nsig/ntot)'] = dfw['n_sig_p<0.05'].astype(int).astype(str) + '/' + dfw['n_subjects'].astype(int).astype(str)
    dfw['KS p (D)'] = dfw.apply(lambda r: f"{r['KS_p']:.3f} ({r['KS_D']:.3f})", axis=1)
    dfw['Binom tail p'] = dfw['binom_tail_p'].round(3)
    out = dfw[['group', 'Cond', 'p_X2 < 0.05 (nsig/ntot)', 'KS p (D)', 'Binom tail p']].rename(columns={'group': 'Group', 'Cond': 'Cond'})
    # collapse Group and cond into group, cond column
    out['Group, Cond'] = out['Group'] + ', ' + out['Cond']
    out = out.drop(columns=['Group', 'Cond'])
    return out.set_index(['Group, Cond']).sort_index()

pre_compact  = make_compact_table(group_summary_df, 'Pre')
post_compact = make_compact_table(group_summary_df, 'Post')

print("# Compact Pre-window summary")
print(pre_compact.to_csv(sep=';', float_format='%.6g'))
display(pre_compact)
save_table_csv(pre_compact, 'chi2_pre_compact.csv')


print("# Compact Post-window summary")
print(post_compact.to_csv(sep=';', float_format='%.6g'))
display(post_compact)
save_table_csv(post_compact, 'chi2_post_compact.csv')





# %% [markdown]
# ### MBP variation modeling

# %% [markdown]
# ### Model driven: Huber 2022 fit

# %% [markdown]
# #### Definition

# %%
# model driven: Huber2022 definition of model, guess, bounds callables

def mbp_model_Huber2022(t, h0, h1, k1, k2, k3, k4, k5, k6, k7):
    """
    Huber2022 model for MBP fitting on the time vector t with the function:
    M(t) =  R(t){t>0} 
          + S(t){t<=0}
    
    where:
    
    R(t) = k1 + k2 * (1 / (1 + exp(-(t - k3)/k4))) + k5 * exp(-((t - k6) ** 2) / (2 * k7 ** 2))
    S(t) = h0 + h1 * t
    
    to be used with scipy.optimize.curve_fit.
    """
    def R(t): # t>0
        return (k1 # baseline
                + k2 * (1 / (1 + np.exp(-(t - k3)/k4))) # sigmoidal
                + k5 * np.exp(-((t - k6) ** 2) / (2 * k7 ** 2)) # gaussian
                )
    def S(t): # t<=0
        return h0 + h1 * t
    
    R_t = R(t[t>0])
    S_t = S(t[t<=0])
    M_t = np.concatenate((S_t, R_t))
    
    return M_t

def mbp_guess_Huber2022(mbp_null,period):
    # begin by using a constant signal, but initialize values with 
    # close-to-expected values
    h0 = mbp_null # constant level for linear
    h1 = 0 # no slope for linear
    k1 = mbp_null # offset of sig/gau
    k2 = 0 # zero amp for sigmoid
    k3 = (period[1]+period[0])/2 + 1*(period[1]+period[0])/6 # sigmoid center
    k4 = 0.25 # sigmoid steepness factor, in t=k3 la derivata è k2/4*k4
    k5 = 0 # Gaussian amplitude
    k6 = (period[1]+period[0])/2 + 2*(period[1]+period[0])/6 # Gaussian center
    k7 = (period[1]-period[0]) / 6 # Gaussian sigma
    return h0, h1, k1, k2, k3, k4, k5, k6, k7
    
def mbp_bound_Huber2022(mbp_null,period):
    b_h0 = (-np.inf,np.inf) # intercept t = 0
    b_h1 = (-np.inf,0) # slope 
    b_k1 = (-np.inf,np.inf) # offset of sig/gau 
    b_k2 = (-np.inf,0.25) # sigmoid amp
    b_k3 = (0,period[1]) # center of sigmoid
    # k3 is 0 to period[1], we don't want exp(-(t-k3)/k4) to overflow!
    # so k4 must be positive and exp(-(t-k3)/k4) << exp(700) -> k4 > (k3 - t)/70 
    # which brings us to k4 > (period[1]-0)/70 = period[1]/70
    b_k4 = (period[1]/70,np.inf) # sigmoid steepness factor
    b_k5 = (0,0.25) # gaussian amp
    b_k6 = (0,period[1]+(period[1]-period[0])/2) # center of gaussian
    b_k7 = (0,(period[1]-period[0])) # gaussian sigma
    b_ = np.array([b_h0,b_h1,b_k1,b_k2,b_k3,b_k4,b_k5,b_k6,b_k7])
    return (b_.T[0],b_.T[1])

mbp_names_Huber2022 = ['h0','h1','k1','k2','k3','k4','k5','k6','k7']


plt.close('all')
xvals = np.arange(TMIN*2, TMAX*2, 0.01)
# plot model for reference
h0 = 0.05
h1 = -0.05/0.5
k1 = 0.05
k2 = k5 = 0.025


k4 = k7 = 0.1 # sigmoid steepness and gaussian sigma
k3s = [0.15,0.35,0.45]
k6s = [0.20, 0.40, 0.50]

# define colors using seaborn, start from dark, go to light from len(k3s) different colors
colors = sns.color_palette("crest", len(k3s))
fig,axs = plt.subplots(3,1, figsize=(3.315, 3.5), sharex=True,sharey=True)
for i, (k3, ax) in enumerate(zip(k3s, axs.flatten())):
    linestyles = ['-','--',':']
    for j, k6 in enumerate(k6s):
        ax.plot(xvals, mbp_model_Huber2022(xvals,  
                                           h0, h1, k1, k2, k3, k4, k5, k6, k7), 
                label=f'k6={k6:.2f} s', #color=thesecolors[j]
                color='k',
                linestyle=linestyles[j]
                )
    
    ax.axhline(0.05, color='0.5', linewidth=0.7, linestyle='--')
    ax.set_ylim(0.025, 0.15)
    ax.set_xlim(TMIN*2, TMAX*2)
    # vgrid each 0.5, plus insert thick line from -0.5 to 0.5 annotated with "T"
    # vertical grid every 0.5s and minor every 0.25s
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(True, axis='x', which='major', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

    # thick horizontal bar from -0.5 to 0.5 labeled "T"
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    ybar = y0 #- 0.05 * yr  # place slightly below top
    bar_thickness = max(2.5, 0.02 * (ax.bbox.height))  # sensible linewidth in display coords
    ax.hlines(ybar, -0.5, 0.5, colors='k', linewidth=4, zorder=25)
    ax.annotate('T', xy=(0.45, ybar + 0.02*1e-2 * yr), xytext=(0, 0), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=30)
    # add lowercase panel labels with str(ord('a')+i)
    # add lowercase panel labels
    panel_label = chr(ord('a') + i)
    ax.text(0.02, 1.1, f"({panel_label}) k3={k3}", transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

#plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'neuwirt.svg'))
plt.savefig(os.path.join(FIGURES_DIR, 'neuwirt.eps'))
plt.show(block=False)

# %% [markdown]
# #### Fitting & plotting

# %%
# fit Huber
subjects = sorted(df_mbp['Subject'].unique())

# Fit all subjects using Huber2022 model
df_pfit_Huber2022 = fit_allsubjects(df_mbp, subjects, condition, 
                          "Time","MBP",null_mbp(B=NBIN),
                          mbp_model_Huber2022,
                          mbp_guess_Huber2022,
                          mbp_bound_Huber2022,
                          mbp_names_Huber2022,
                          interpolate_dt=INTP_DT)

print("Fitted parameters for Huber2022 model:")
display(df_pfit_Huber2022)
save_table_csv(df_pfit_Huber2022, 'fitparams_huber2022.csv')


# %%
# Plot all subjects with fits
plot_mbpfit_allsubjects(df_mbp, df_pfit_Huber2022, subjects, condition, 
                        x = "Time",
                        y = "MBP", y0 = "E(MBP_0)", vary0 = "Var(MBP_0)",
                        model = mbp_model_Huber2022, 
                        names = mbp_names_Huber2022, 
                        )


pdesc_Huber2022 = {
    'h0': 'Linear intercept',
    'h1': 'Linear slope',
    'k1': 'Sig/Gau offset',
    'k2': 'Sigmoid amplitude',
    'k3': 'Sigmoid center',
    'k4': 'Sigmoid steepness',
    'k5': 'Gaussian amplitude',
    'k6': 'Gaussian center',
    'k7': 'Gaussian sigma',
}

plot_mbpfit_parameters_group(df_pfit_Huber2022, subj_to_group, param_interpretation=pdesc_Huber2022)

fig = plot_mbpfit_fitted_group(df_pfit_Huber2022, subj_to_group, condition, 
                               mbp_model_Huber2022, mbp_names_Huber2022, "MBP")

# %% [markdown]
# ### Model driven: Neuwirt (2001) sawtooth 

# %% [markdown]
# Sawtooth waveform from Neuwirth, E. (2001). Designing a Pleasing Sound Mathematically. Mathematics Magazine, 74(2), 91–98. https://doi.org/10.1080/0025570X.2001.11953044
# 
# We define a the corresponding phase of time and time shift as angles with the period:
# 
# $$\phi := \frac{2\pi}{T}t, \ \theta := \frac{2\pi}{T}\tau$$
# 
# then 
# 
# $$N(t) = A \cdot \frac{\sin(\phi-\theta))}{1+q^2-2q\cos(\phi-\theta)}$$
# 
# A controls the maximum amplitude of the wave, q is a shape parameter and tau is the time point representing the mid-rise instant
# 
# Important note on q:
# 
#  - q < -1 leads to a symmetric pair of negative-positive peaks around tau
#  - q = -1 leads to degeneration into tangent function
#  - -1 < q < 0 leads to near-linear increase centered in tau, negative bump
#  - q = 0 is a simple sine wave --> our starting point, we use $\tau = 0.25$ so that its minimum falls into t = 0
#  - 0 < q < 1 leads to near-linear decrease followed by positive bump rising in tau
#  - q = 1 leads to $cot(\phi-\theta)/2$
# 
# Starting guess: sinusoid, suppression synchronized to t = 0 as seen by frequency analysis
#  - 0 <= A < 1 --> guess: 1
#  - 0 <= q < 1 --> guess: 0
#  - -0.5 < tau < 0.5 --> guess: 0.25

# %% [markdown]
# #### Definition

# %%
# model driven: Neuwirt2001 definition of model, guess, bounds callables
def mbp_model_Neuwirt2001(t, #k, 
                          A, q, tau, period=1.0):
    """
    Neuwirt2001 curve ("pleasant sound") for MBP fitting on a phase vector phi
    N(t) =  k + A * sin(phi - theta) / (1 + q*2 - 2*q*cos(phi - theta))

    where we assume that phi and theta correspond already to t and tau (shift in time)
    scaled by the period so that:
    phi = 2pi * t / period
    theta = 2pi * tau / period


    to be used with scipy.optimize.curve_fit
    """
    phi = 2 * np.pi * t / period
    theta = 2 * np.pi * tau / period

    return 0.10 + A * np.sin(phi - theta) / (1 + q**2 - 2*q*np.cos(phi - theta))#k + A * np.sin(phi - theta) / (1 + q**2 - 2*q*np.cos(phi - theta))
    
def mbp_guess_Neuwirt2001(mbp_null,period):
    # begin by using a sinusoidal 
    #k = mbp_null
    A = mbp_null # amplitude
    q = 0 # damping/shape factor
    tau = 0#(period[1] - period[0]) / 4 # phase shift
    return  A, q, tau#k, A, q, tau

def mbp_bound_Neuwirt2001(mbp_null,period):
    #b_k = (0, 0.2) # baseline
    b_A = (-1e-15,0.20) # amplitude
    b_q = (-1e-15,0.95) # damping/shape factor: must be < 1 to avoid singularities (becomes cotangent)
    b_tau = (period[0], period[1]) # phase shift
    b_ = np.array([#b_k,
                   b_A,b_q,b_tau])
    return (b_.T[0],b_.T[1])

mbp_names_Neuwirt2001 = [#'k',
                         'A','q','tau']

plt.close('all')
xvals = np.arange(TMIN*2, TMAX*2, 0.01)
# plot Neuwirt2001 model for reference, with A=1 and varying theta = pi/4, pi/2, 3pi/4 with three similar colors
# and q = 0, q = 0.1, q = 0.2, q = 0.5 in different plots
qs = [0, 0.2, 0.5]
taus = [0, 0.2, 0.4] # in seconds
# define colors using seaborn, start from dark, go to light from len(qs) different colors
colors = sns.color_palette("crest", len(qs))
fig,axs = plt.subplots(3,1, figsize=(3.315, 3.5), sharex=True,sharey=True)
for i, (q, ax) in enumerate(zip(qs, axs.flatten())):
    linestyles = ['-','--',':']
    for j, tau in enumerate(taus):
        ax.plot(xvals, mbp_model_Neuwirt2001(xvals, #0, 
                                         1, q, tau)-0.1, 
                label=f'tau={tau:.2f} s', #color=thesecolors[j]
                color='k',
                linestyle=linestyles[j]
                )
    #ax.set_title(f'Neuwirt2001 Model for MBP with q={q} (k=0, A=1)')
    #ax.legend()
    # set yticks only at 0, +- 0.5, +- 1 and call them 0, A/2, A
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(['-A', '-A/2', '0', 'A/2', 'A'])
    ax.axhline(0, color='0.5', linewidth=0.7, linestyle='--')
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlim(TMIN*2, TMAX*2)
    # vgrid each 0.5, plus insert thick line from -0.5 to 0.5 annotated with "T"
    # vertical grid every 0.5s and minor every 0.25s
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(True, axis='x', which='major', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

    # thick horizontal bar from -0.5 to 0.5 labeled "T"
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    ybar = y0 #- 0.05 * yr  # place slightly below top
    bar_thickness = max(2.5, 0.02 * (ax.bbox.height))  # sensible linewidth in display coords
    ax.hlines(ybar, -0.5, 0.5, colors='k', linewidth=4, zorder=25)
    ax.annotate('T', xy=(0.45, ybar + 0.02 * yr), xytext=(0, 0), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=30)
    # add lowercase panel labels with str(ord('a')+i)
    # add lowercase panel labels
    panel_label = chr(ord('a') + i)
    ax.text(0.02, 1.1, f"({panel_label}) q={q}", transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

#plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'neuwirt.svg'))
plt.savefig(os.path.join(FIGURES_DIR, 'neuwirt.eps'))
plt.show(block=False)


# %%
# fit Neuwirt2001
T  = 1.0 # period for circular modeling
subjects = sorted(df_mbp['Subject'].unique())

# Fit all subjects using Neuwirt2001 model
df_pfit_Neuwirt2001 = fit_allsubjects(df_mbp, subjects, condition, 
                          "Time","MBP",null_mbp(B=NBIN),
                          mbp_model_Neuwirt2001,
                          mbp_guess_Neuwirt2001,
                          mbp_bound_Neuwirt2001,
                          mbp_names_Neuwirt2001,
                          interpolate_dt=INTP_DT)

print("Fitted parameters for Neuwirt2001 model:")
display(df_pfit_Neuwirt2001)
save_table_csv(df_pfit_Neuwirt2001, 'fitparams_neuwirt2001.csv')

# Plot all subjects with fits
figall = plot_mbpfit_allsubjects(df_mbp, df_pfit_Neuwirt2001, subjects, condition, 
                        x = "Time",
                        y = "MBP", y0 = "E(MBP_0)", vary0 = "Var(MBP_0)",
                        model = mbp_model_Neuwirt2001, 
                        names = mbp_names_Neuwirt2001,
                        )
figall.set_size_inches(5.31, 1.5 * len(subjects))
plt.show(block=False)

pdesc_Neuwirt2001 = {
    #'k': 'Baseline',
    'A': 'Amplitude',
    'q': 'Damping/shape factor',
    'theta': 'Phase shift (rad)',
}

_=plot_mbpfit_parameters_group(df_pfit_Neuwirt2001, subj_to_group, 
                             param_interpretation=pdesc_Neuwirt2001)


# %%
plt.close('all')
fig_fit,axes = plt.subplots(4,2, figsize=(5.31, 5.31), sharex=True,sharey=True)

_ = plot_mbpfit_fitted_group(df_pfit_Neuwirt2001, subj_to_group, condition, 
                               mbp_model_Neuwirt2001, mbp_names_Neuwirt2001, "MBP",
                               axes=axes[:,0])



# again, but tau=0 --> we see only the effect of amplitude, shape, k
df_pfit_Neuwirt2001_sametau = df_pfit_Neuwirt2001.copy()
df_pfit_Neuwirt2001_sametau['tau'] = 0.0
_ = plot_mbpfit_fitted_group(df_pfit_Neuwirt2001_sametau, subj_to_group, condition, 
                               mbp_model_Neuwirt2001, mbp_names_Neuwirt2001, "MBP",
                               axes=axes[:,1], add_stim=False)

for ax in axes.flatten():
    ax.set_title('')

for ax,ttl in zip(axes[:,0],['MBP','MBP (HC)','MBP (eMCS)','MBP (pDoC)']):
    ax.set_ylabel(ttl, fontsize=10)
    
for ax in axes[:,1]:
    ybar = add_stim_to_ax(ax, icon = 'τ',bar_color='w',bar_alpha=0)
    fs = 10
    # add left triangle marker
    ax.annotate('◄', xy=(0.025, ybar), xytext=(0, -fs//2+1), textcoords='offset points',
                ha='center', va='bottom', fontsize=fs,
                fontname='DejaVu Sans',zorder=20)  # font supporting utf-8 icons

# axes[0,0].set_title('$\\tau$ fitted', fontsize=10)
# axes[0,1].set_title('$\\tau=0$', fontsize=10)
axes[0,0].set_title('')
axes[0,1].set_title('')

axes[-1,0].set_xlabel('Time (s)', fontsize=10)
axes[-1,1].set_xlabel('Time + $\\tau$ (s)', fontsize=10)

fig_fit.savefig(os.path.join(FIGURES_DIR, 'mbpfit_neuwirt2021_group.svg'), bbox_inches='tight')
fig_fit.savefig(os.path.join(FIGURES_DIR, 'mbpfit_neuwirt2021_group.eps'), bbox_inches='tight')
plt.show(block=False)

# %%


pfit_df = df_pfit_Neuwirt2001.copy()
subject = 'sABI_0016'
condition = 'ODDBALL'
names = mbp_names_Neuwirt2001
model = mbp_model_Neuwirt2001

pfits2 = {'sABI_0019': [0.023, 0.7, -0.5],
          'HC_0001': [0.045, 0.5, 0.35],}

# Keep backward compatibility with legacy IDs, but run safely on current IDs (e.g., sub-XX)
if subject not in pfit_df.index:
    subject = str(pfit_df.index[0])
    print(f"Requested demo subject not found; using '{subject}' instead.")

subj_df = df_mbp.query(f"Subject == '{subject}' and Condition == '{condition}' and Time < {TMAX} and Time > {TMIN}").copy()
time_data = subj_df['Time'].to_numpy()
mbp_data = subj_df['MBP'].to_numpy()

pfit = pfit_df.loc[subject, names].to_numpy()
yfit_at_data = model(time_data, *pfit)

time_intp = np.arange(TMIN, TMAX, INTP_DT)
mbp_intp = circ_interp(time_data, mbp_data, time_intp)
yfit_intp = model(time_intp, *pfit)


fig,ax = plt.subplots(1,1, figsize=(3.315, 2.5))
ax.scatter(time_data, mbp_data, label='Data', color='blue', marker='x')
ax.scatter(time_data, yfit_at_data, marker='+',label='Fit at data', color='red')

ax.plot(time_intp, mbp_intp, label='Upsampled MBP', color='cyan', alpha=0.5)
ax.plot(time_intp, yfit_intp, label='Fit (upsampled)', color='orange')


def rmse(ytrue, ypred):
    return np.sqrt(np.mean((ytrue - ypred)**2))

rmse0 = rmse(mbp_data, 0.10)
rmse0i = rmse(mbp_intp, 0.10)
rmse1 = rmse(mbp_data, yfit_at_data)
rmse1i = rmse(mbp_intp, yfit_intp)

# annotate on the plot with the same colors
ax.annotate(f'RMSE0: {rmse0:.2g} / {rmse0i:.2g}', xy=(0.95, 0.9), xycoords='axes fraction',
            ha='right', va='top', color='blue', fontsize=9)
ax.annotate(f'RMSE1: {rmse1:.2g} / {rmse1i:.2g}', xy=(0.95, 0.825), xycoords='axes fraction',
            ha='right', va='top', color='red', fontsize=9)
print("Rsquared1 on data:", 1 - (rmse1**2) / (rmse0**2))
print("RRsquared1 on intp:", 1 - (rmse1i**2) / (rmse0i**2))

pfit2 = pfits2.get(subject,None)
if pfit2 is not None:
    yfit2_at_data = model(time_data, *pfit2)
    yfit2_intp = model(time_intp, *pfit2)
    ax.scatter(time_data, yfit2_at_data, marker='o',label='Example fit at data', color='green', facecolors='none')
    ax.plot(time_intp, yfit2_intp, label='Example fit (upsampled)', color='green', linestyle='--')
    rmse2 = rmse(mbp_data, yfit2_at_data)
    rmse2i = rmse(mbp_intp, yfit2_intp)
    ax.annotate(f'RMSE2: {rmse2:.2g} / {rmse2i:.2g}', xy=(0.95, 0.75), xycoords='axes fraction',
                ha='right', va='top', color='green', fontsize=9)
    print("Rsquared2 on data:", 1 - (rmse2**2) / (rmse0**2))
    print("RRsquared2 on intp:", 1 - (rmse2i**2) / (rmse0i**2))

ax.set_ylim(0, 0.2)
finalize_mbp_axes(ax)
plt.show(block=False)

# compute rmse0, rmse1, rsquared1 for all subjects and display as dataframe
rmse_rows = []
for subject in subjects:
    condition = 'ODDBALL'
    subj_df = df_mbp.query(f"Subject == '{subject}' and Condition == '{condition}' and Time < {TMAX} and Time > {TMIN}").copy()
    time_data = subj_df['Time'].to_numpy()
    mbp_data = subj_df['MBP'].to_numpy()

    pfit = pfit_df.loc[subject, names].to_numpy()
    yfit_at_data = model(time_data, *pfit)

    time_intp = np.arange(TMIN, TMAX, INTP_DT)
    mbp_intp = circ_interp(time_data, mbp_data, time_intp)
    yfit_intp = model(time_intp, *pfit)

    rmse0 = rmse(mbp_data, 0.10)
    rmse0i = rmse(mbp_intp, 0.10)
    rmse1 = rmse(mbp_data, yfit_at_data)
    rmse1i = rmse(mbp_intp, yfit_intp)

    rsq1 = 1 - (rmse1**2) / (rmse0**2)
    rsq1i = 1 - (rmse1i**2) / (rmse0i**2)

    rmse_rows.append({
        'Subject': subject,
        'RMSE0': rmse0,
        'RMSE0_intp': rmse0i,
        'RMSE1': rmse1,
        'RMSE1_intp': rmse1i,
        'Rsquared1': rsq1,
        'Rsquared1_intp': rsq1i
    })
rmse_df = pd.DataFrame(rmse_rows)
print("RMSE and Rsquared summary for all subjects:")
display(rmse_df)
save_table_csv(rmse_df, 'rmse_rsquared_subjects.csv')

# this time wrap RMSE_0, RMSE, Rsquared computation in a function taking df_mbp, pfit_df, model, names, condition
def compute_rmse_rsquared(df_mbp, pfit_df, model, names, condition, tmin, tmax, intp_dt):
    subjects = sorted(df_mbp['Subject'].unique())
    rmse_rows = []
    for subject in subjects:
        subj_df = df_mbp.query(f"Subject == '{subject}' and Condition == '{condition}' and Time < {tmax} and Time > {tmin}").copy()
        time_data = subj_df['Time'].to_numpy()
        mbp_data = subj_df['MBP'].to_numpy()

        pfit = pfit_df.loc[subject, names].to_numpy()
        yfit_at_data = model(time_data, *pfit)

        time_intp = np.arange(tmin, tmax, intp_dt)
        mbp_intp = circ_interp(time_data, mbp_data, time_intp)
        yfit_intp = model(time_intp, *pfit)

        rmse0 = rmse(mbp_data, 0.10)
        rmse0i = rmse(mbp_intp, 0.10)
        rmse1 = rmse(mbp_data, yfit_at_data)
        rmse1i = rmse(mbp_intp, yfit_intp)

        rsq1 = 1 - (rmse1**2) / (rmse0**2)
        rsq1i = 1 - (rmse1i**2) / (rmse0i**2)

        rmse_rows.append({
            'Subject': subject,
            'RMSE0': rmse0,
            'RMSE0_intp': rmse0i,
            'RMSE1': rmse1,
            'RMSE1_intp': rmse1i,
            'Rsquared1': rsq1,
            'Rsquared1_intp': rsq1i
        })
    return pd.DataFrame(rmse_rows)

df_quality = compute_rmse_rsquared(df_mbp, df_pfit_Neuwirt2001, mbp_model_Neuwirt2001, mbp_names_Neuwirt2001, 'ODDBALL', TMIN, TMAX, INTP_DT)
display(df_quality)
save_table_csv(df_quality, 'quality_interpolated.csv')


# %%
# add neuwirt delta to df_pfit_Neuwirt2001 for boxplot
def neuwirt_delta(A,q):
    delta = A * (1+q**2)*np.sqrt(1-4*q**2/(1+q**2)**2)/(1 - q**2)**2
    return delta

neuwirt4boxplot = df_pfit_Neuwirt2001.copy()
neuwirt4boxplot['$\\Delta$'] = neuwirt4boxplot.apply(
    lambda row: pd.Series(neuwirt_delta(row['A'], row['q'])),
    axis=1
)

def compute_rmse_rsquared(df_mbp, pfit_df, model, names, condition, tmin, tmax):
    subjects = sorted(df_mbp['Subject'].unique())
    rmse_rows = []
    for subject in subjects:
        subj_df = df_mbp.query(f"Subject == '{subject}' and Condition == '{condition}' and Time < {tmax} and Time > {tmin}").copy()
        time_data = subj_df['Time'].to_numpy()
        mbp_data = subj_df['MBP'].to_numpy()

        pfit = pfit_df.loc[subject, names].to_numpy()
        yfit_at_data = model(time_data, *pfit)

        rmse0 = rmse(mbp_data, 0.10) # can use also the mean of mbp_data since by construction they sum up to 1
        rmse1 = rmse(mbp_data, yfit_at_data)

        rsq1 = 1 - (rmse1**2) / (rmse0**2)

        rmse_rows.append({
            'Subject': subject,
            '$RMSE_0$': rmse0,
            '$RMSE$': rmse1,
            '$R^2$': rsq1,
        })
        retdf = pd.DataFrame(rmse_rows)
        retdf.set_index('Subject', inplace=True)
    return retdf

df_quality = compute_rmse_rsquared(df_mbp, df_pfit_Neuwirt2001, 
                                   mbp_model_Neuwirt2001, mbp_names_Neuwirt2001, 
                                   'ODDBALL', TMIN, TMAX)
# merge on index
neuwirt4boxplot = neuwirt4boxplot.merge(df_quality, left_index=True, right_index=True)
save_table_csv(df_quality, 'quality_noninterpolated.csv')


# add group and clinical diagnosis info
df_subjects = load_subject_metadata(res_folder, data_folder)
df_clindiag = df_subjects[['Group','Subject','Clinical diagnosis']].copy()
# use group to fill missing in clinical diagnosis
df_clindiag.loc[df_clindiag['Clinical diagnosis'].isna(), 'Clinical diagnosis'] = df_clindiag.loc[df_clindiag['Clinical diagnosis'].isna(), 'Group']
# to subject -> diagnosis mapping
subj_to_diagnosis = df_clindiag.set_index('Subject')['Clinical diagnosis'].to_dict()
# add columns
neuwirt4boxplot.reset_index(names='Subject', inplace=True)
neuwirt4boxplot['Group'] = neuwirt4boxplot['Subject'].map(subj_to_group)
neuwirt4boxplot['Clinical diagnosis'] = neuwirt4boxplot['Subject'].map(subj_to_diagnosis)


neuwirt4boxplot.to_csv(os.path.join(CACHE_DIR, 'neuwirt4boxplot.csv'), index=False)

# MEDIAN AND CI TABLE
order = [#'k', 
         'A', '$\\Delta$', 'q', 'tau','$R^2$','$RMSE$']
params = [p for p in order if p in neuwirt4boxplot.columns]
groups = sorted(neuwirt4boxplot['Group'].unique())
data = {g: [] for g in groups}
for param in params:
    for g in groups:
        arr = neuwirt4boxplot.loc[neuwirt4boxplot['Group'] == g, param].dropna().to_numpy()
        if arr.size == 0:
            data[g].append(np.nan)
        else:
            med = np.median(arr)
            low, high = np.percentile(arr, [2.5, 97.5])
            # format A and Delta as percentages
            if param in ('A', '$\\Delta$','$RMSE$'):
                data[g].append(f"{med*100:.1f}% [{low*100:.1f}%, {high*100:.1f}%]")
            else:
                data[g].append(f"{med:.3f} [{low:.3f}, {high:.3f}]")

table_neuwirt = pd.DataFrame(data, index=params)
display(table_neuwirt)
# print as csv with ; separator
print(table_neuwirt.to_csv(sep=';', float_format='%.6g'))
save_table_csv(table_neuwirt, 'table_neuwirt_median_ci.csv')

# STATISTICAL TESTING
from scipy import stats
import scikit_posthocs as skph
import warnings

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

neuwirt4boxplot['Condition'] = 'ODDBALL'  # single condition for all
stats_neuwirt = run_stat_tests_ncheck(neuwirt4boxplot, 
                                 condition_col='Condition', 
                                 group_col='Group', 
                                 alpha=0.05, 
                                 p_adjust='holm',
                                 group_comparisons=[('HC', 'eMCS'), ('eMCS', 'pDoC'), ('HC', 'pDoC')])

# display and ;-csv .3g float format export three sections: median-iqr, normality, kruskal-wallis/conover-iman
# stats_neuwirt_median = stats_neuwirt.loc[:, ('Median (IQR)', slice(None))]
# print("Median and IQR results:")
# display(stats_neuwirt_median)
# print(stats_neuwirt_median.to_csv(sep=';', float_format='%.3g'))

# normality
stats_neuwirt_normality = stats_neuwirt.loc[:, ('Normality Test', slice(None))]
print("Normality test results (Shapiro-Wilk p-values):")
display(stats_neuwirt_normality)
print(stats_neuwirt_normality.to_csv(sep=';', float_format='%.3g'))
save_table_csv(stats_neuwirt_normality, 'stats_neuwirt_normality.csv', float_format='%.3g')

# kruskal-wallis and conover-iman
stats_neuwirt_kw = stats_neuwirt.loc[:, ('Kruskal-Wallis', slice(None))]
stats_neuwirt_ci = stats_neuwirt.loc[:, ('Conover', slice(None))]
stats_neuwirt_kwci = pd.concat([stats_neuwirt_kw, stats_neuwirt_ci], axis=1)
print("Kruskal-Wallis and Conover-Iman post-hoc test results:")
display(stats_neuwirt_kwci)
print(stats_neuwirt_kwci.to_csv(sep=';', float_format='%.3g'))
save_table_csv(stats_neuwirt_kwci, 'stats_neuwirt_kw_conover.csv', float_format='%.3g')


# BOXPLOTS WITH OVERLAID POINTS AND STATISTICS
from statannotations.Annotator import Annotator
from matplotlib.lines import Line2D

cols2plot = np.array([['A','$\\Delta$'],['tau','q'],['$R^2$','$RMSE$']])

fig, axs = plt.subplots(*cols2plot.shape, figsize=(5.31,6))

for ax, col in zip(axs.flatten(), cols2plot.flatten(), strict=True):
    
    sns.boxplot(
        data=neuwirt4boxplot, x='Group', y=col, hue='Group',
        showfliers=False,
        palette={'HC': C_HC, 'eMCS': C_EMCS, 'pDoC': C_PDOC},
        ax=ax
    )

    group_order = ['HC', 'eMCS', 'pDoC']
    # get unique clinical diagnoses per group
    diags_in_group = neuwirt4boxplot.groupby('Group')['Clinical diagnosis'].unique()
    diags = sorted(neuwirt4boxplot['Clinical diagnosis'].dropna().unique())
    marker_cycle = ['o','X','^','D','P','s','*','v','<','>','h','H']
    diag_to_marker = {d: marker_cycle[j % len(marker_cycle)] for j, d in enumerate(diags)}
    
    width = 0.18  # displacement width

    for i, (group, group_diags) in enumerate(diags_in_group.items()):
        print(f"Group {group} has diagnoses: {group_diags}")
        group_data = neuwirt4boxplot[neuwirt4boxplot['Group'] == group]

        n_diag = len(group_diags)
        if n_diag == 0:
            continue
        # Center the swarms symmetrically around the box
        # Calculate offsets: center at 0 if odd, distribute symmetrically if even
        if n_diag == 1:
            offsets = [0]
        else:
            # For even n_diag, distribute symmetrically around 0
            offsets = np.linspace(-width/2, width/2, n_diag)
        for j, diag in enumerate(group_diags):
            diag_data = group_data[group_data['Clinical diagnosis'] == diag]
            if diag_data.empty:
                continue
            x_pos = i + offsets[j]
            sc = ax.scatter(
                np.full(len(diag_data), x_pos),
                diag_data[col],
                color='0.6',
                s=15,
                marker=diag_to_marker[diag],
                label=diag if i == 0 else None,  # only label once for legend
                zorder=10,
                alpha=1
            ) 
            offsetsx,offsetsy = sc.get_offsets().T
            jitter = 0
            yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
            # scan offsets: if one offset is close to the previous vertically (0.05*yrange), 
            # shift it horizontally by small random jitter to the right
            for idx in range(1, len(offsetsx)):
                if abs(offsetsy[idx] - offsetsy[idx-1]) < 0.075 * yrange:
                    jitter += 1
                    
                    offsetsx[idx] += np.clip(jitter*0.2*width*(-1)**(jitter%2), -width/2, width/2)
                else:
                    jitter = 0
            sc.set_offsets(np.c_[offsetsx, offsetsy])

    # Use the same Holm-corrected Conover-Iman p-values exported above.
    pairs = [('HC','eMCS'), ('HC','pDoC'), ('eMCS','pDoC')]
    posthoc_data = neuwirt4boxplot[['Group', col]].dropna()
    group_values = [
        posthoc_data.loc[posthoc_data['Group'] == group, col]
        for group in group_order
    ]
    _, omnibus_p = stats.kruskal(*group_values)

    if omnibus_p < ALPHA:
        conover_holm = skph.posthoc_conover(
            posthoc_data,
            val_col=col,
            group_col='Group',
            p_adjust='holm',
        )
        corrected_pvalues = [
            conover_holm.loc[group1, group2]
            for group1, group2 in pairs
        ]
    else:
        corrected_pvalues = [1.0] * len(pairs)

    annot = Annotator(ax, pairs, data=neuwirt4boxplot, x='Group', y=col)
    annot.configure(text_format='star', loc='inside',
                    hide_non_significant=True, verbose=0)
    annot.set_pvalues_and_annotate(corrected_pvalues)

    # Marker-shape legend for diagnosis (all unique diags)
    all_diags = sorted(neuwirt4boxplot['Clinical diagnosis'].dropna().unique())
    marker_cycle = ['o','s','^','D','P','X','*','v','<','>','h','H']
    diag_to_marker = {d: marker_cycle[j % len(marker_cycle)] for j, d in enumerate(all_diags)}
    marker_handles = [
        Line2D([0], [0], marker=m, color='0.2', linestyle='', markersize=6, label=d)
        for d, m in diag_to_marker.items()
    ]
    
    # changes specific to each plot
    if col == 'A':
        ax.set_ylabel('A', fontsize=10)
        # yticks in percentage
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{y*100:.0f}%" for y in yticks])
    elif col == '$\\Delta$':
        ax.set_ylabel('$\\Delta$', fontsize=10)
        # yticks in percentage
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{y*100:.0f}%" for y in yticks])
        # sharey with A plot
        ax.sharey(axs.flatten()[cols2plot.flatten().tolist().index('A')])
    elif col == '$R^2$':
        ax.set_ylabel('$R^2$', fontsize=10)
        ax.set_ylim(min(-0.1, ax.get_ylim()[0]), max(1.05, ax.get_ylim()[1]))
    elif col == '$RMSE$':
        ax.set_ylabel('RMSE', fontsize=10)
        # yticks in percentage
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{y*100:.0f}%" for y in yticks])
        ax.set_ylim(0, max(0.1, ax.get_ylim()[1]))
    elif col == 'tau':
        ax.set_ylabel('$\\tau$ (s)', fontsize=10)
        # set sym ylim
        ylim = ax.get_ylim()
        ax.set_ylim(-max(abs(ylim[0]), abs(ylim[1])), max(abs(ylim[0]), abs(ylim[1])))
    elif col == 'q':
        ax.set_ylabel('q', fontsize=10)
        ax.set_ylim(min(0, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1]))
    
    #ax.legend(handles=marker_handles, title='Clinical diagnosis', loc='upper right')

    #ax.set_title(f'Fitted parameter: {col}')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mbpfit_neuwirt2021_boxplots.svg'), bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'mbpfit_neuwirt2021_boxplots.eps'), bbox_inches='tight')
plt.show(block=False)

# %%
# Neuwirt model phase on time courses
groups = df_mbp['Group'].unique()
fig, axs = plt.subplots(len(groups),1, figsize=(12, 2*len(groups)), sharex=True, sharey=True)

group2color = {
    'HC':  C_HC,
    'eMCS':C_EMCS,
    'pDoC':C_PDOC,
}
for ax, group in zip(axs, groups, strict=True):
    group_data = df_mbp[df_mbp['Group'] == group]
    sns.lineplot(data=group_data, x='Time', y="MBP", hue='Group',
                 palette=group2color, alpha=0.5,
                 ax=ax, legend=False)
    ax.set_title(f'MBP per Subject in group {group}, data: {"MBP"}')
    finalize_mbp_axes(ax,hline=df_mbp["E(MBP_0)"].mean())
    add_stim_to_ax(ax)
    ax.set_xlim(TMIN, TMAX)
    
    # plot vertical lines at phase times from neuwirt model
    group_pfit = df_pfit_Neuwirt2001[df_pfit_Neuwirt2001.index.isin([s for s in subjects if subj_to_group.get(s) == group])]
    for _, row in group_pfit.iterrows():
        phase_time = row['tau']  # tau is already in seconds for Neuwirt model
        ax.axvline(phase_time, color='blue', linestyle='--', alpha=0.7)
        
plt.tight_layout()
plt.show(block=False)

# %% Block exec to show plots before killing kernel
plt.show(block=True)
