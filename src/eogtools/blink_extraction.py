import os

import numpy as np
import pandas as pd
import scipy.interpolate
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from eogtools.eog import ignore_runtime_warning, plotter_eog, process_eog
from utils.plotting import __cmap_or_cmap_from_color


def blinks_from_eog(edfpath, 
                    eog_dir, 
                    eog_channel,
                    blink_min_dur=0,
                    find_eog_kwargs=None):

    base = os.path.splitext(os.path.basename(edfpath))[0]
    
    # Compute blinks location on vertical EOG
    try:
        blinks, signals = process_eog(
            edfpath,
            extrema=1,
            eog_channel=eog_channel,
            blink_min_durations=blink_min_dur,
            find_eog_kwargs=find_eog_kwargs
        )
        
        r = f"Found {blinks.shape[0]} blinks."
    except ValueError as e:
        r = f"Error processing {eog_channel}: {e}"
    
    # single print better for multiprocessing
    print(f"\n[{edfpath}] > EOG processing results: " + r)
            
    blinks.to_csv(os.path.join(eog_dir, f'{base}_blinks.csv'))
    signals.to_csv(os.path.join(eog_dir, f'{base}_processing_steps.csv'))
    
    return (os.path.join(eog_dir, f'{base}_blinks.csv'),
            os.path.join(eog_dir, f'{base}_processing_steps.csv'))


@ignore_runtime_warning
def extract_blink_waveforms(base,eog_dir):
    
    # Channel D
    try:
        blinks = pd.read_csv(os.path.join(eog_dir, f'{base}_blinks.csv'))
        signals = pd.read_csv(os.path.join(eog_dir, f'{base}_processing_steps.csv'),
                                                    index_col=0)
        # Compute waveforms (dataframe of #valid_blinks x #samples)
        waveforms = _extract_blink_waveforms(signals['mne_filtered'].to_numpy(),
                                            blinks['Peak'].to_numpy(),
                                            blinks['Onset'].to_numpy(),
                                            blinks['Offset'].to_numpy(),
                                            int(1/np.mean(np.diff(signals.index))),
                                            epoch_half_duration=0.5,
                                            remove_onset=True,)
        # Subset of waveforms with high similarity
        waveforms_sim = waveform_similarity(waveforms).mean(axis=0)
        # Save additional column for blinks dataframe with similarity
        waveforms_sim = pd.DataFrame({'Similarity':waveforms_sim.squeeze()})
    except Exception as e:
        print(f'\nException for {base}, channel D -> {e}')
        waveforms = pd.DataFrame(dtype=np.float64)
        waveforms_sim = pd.DataFrame(dtype=np.float64)
        blinks = pd.DataFrame(dtype=np.float64)

    # --------------- Save  -----------
    os.makedirs(eog_dir, exist_ok=True)

    waveforms.to_csv(os.path.join(eog_dir,f'{base}_waveforms.csv'))
    waveforms_sim.to_csv(os.path.join(eog_dir,f'{base}_similarity.csv'))

    # --------------- Plot waveforms pairing resting and oddball --------------
    # Channel D
    fig = plotter_waveforms(waveforms, waveforms_sim, blinks)
    fig.suptitle(f'{base}')
    
    # --------------- Save waveforms plot -------------------------------------
    fig.savefig(os.path.join(eog_dir, f'{base}_waveforms.png'))
    
    plt.close(fig)
    
    return (os.path.join(eog_dir,f'{base}_waveforms.csv'),
            os.path.join(eog_dir,f'{base}_similarity.csv'),
            os.path.join(eog_dir, f'{base}_waveforms.png'))

def _extract_blink_waveforms( 
    eog_data: np.ndarray, 
    peaks: np.ndarray, 
    onsets: np.ndarray, 
    offsets: np.ndarray, 
    sfreq: float, 
    epoch_half_duration: float = 0.5,
    remove_onset: bool = False
    ) -> pd.DataFrame: 
    """ 
    Extract short blink waveforms around each valid blink (peak). 
    The resulting DataFrame has index = peak time, columns = peak-relative time.
    Parameters
    ----------
    eog_data : np.ndarray
        One-dimensional array of the final EOG signal (already cleaned / filtered).
    peaks : np.ndarray
        Times (seconds) of each blink peak.
    onsets : np.ndarray
        Times (seconds) of each blink onset.
    offsets : np.ndarray
        Times (seconds) of each blink offset.
    sfreq : float
        Sampling frequency.
    epoch_half_dur : float
        Half-duration (in seconds) around peak to extract.
    
    Returns
    -------
    pd.DataFrame
        index=peak time, columns=relative time (seconds about peak).
    """
    if not (len(peaks) == len(onsets) == len(offsets)):
        raise ValueError("peaks, onsets, offsets must have same length.")
    
    full_epoch_len = int(round(epoch_half_duration * sfreq * 2))+1
    times = np.linspace(-epoch_half_duration, 
                        epoch_half_duration, 
                        full_epoch_len, endpoint=True)
    
    if len(peaks) == 0:
        empty_df = pd.DataFrame(columns=times)
        empty_df.index.name = 'Peak_s'
        return empty_df
    
    waveforms = []
    for pk_t, on_t, off_t in zip(peaks, onsets, offsets, strict=True):
        onset_idx = int(np.floor(on_t * sfreq))
        offset_idx = int(np.ceil(off_t * sfreq))
        segment_eog = eog_data[onset_idx: offset_idx + 1]
        times_segment = np.linspace(on_t - pk_t, off_t - pk_t, len(segment_eog), endpoint=True)
    
        # Interpolate to the nominal epoch times
        f_int = scipy.interpolate.interp1d(
            times_segment,
            segment_eog,
            fill_value=np.nan,
            bounds_error=False
        )
        waveform = f_int(times)
        # Remove the onset as a baseline (first non-nan value)
        if remove_onset:
            if np.isnan(waveform).all():
                raise RuntimeError(
                    f"All values are NaN for peak at {pk_t} s going from {on_t} to {off_t} s.")
            first_non_nan = np.where(~np.isnan(waveform))[0][0]
            first_non_nan_50ms = first_non_nan + int(0.05 * sfreq)
            waveform -= np.nanmean(waveform[first_non_nan:first_non_nan_50ms])
            
        waveforms.append(waveform)
    
    wave_df = pd.DataFrame(np.array(waveforms), index=peaks, columns=times)
    wave_df.index.name = 'Peak_s'
    return wave_df


def extract_features_from_waveforms(base, eog_folder):
    """
    Analyze waveforms from blinks data and compute features.
    
    Parameters:
    -----------
    base : str
        Base name of the files to analyze
    eog_folder : str
        Path to the folder containing the EOG data
    """
    
    sub, cond = base.split('_')[:2]
    if 'sub-' in sub:
        sub = sub.replace('sub-','')
    else:
        raise ValueError(f"Subject ID in base '{base}' does not start with 'sub-'.")
    if 'task-' in cond:
        condition = cond.replace('task-','')
    else:
        raise ValueError(f"Condition ID in base '{base}' does not start with 'task-'.")
    # Create Features folder if it doesn't exist
    os.makedirs(eog_folder, exist_ok=True)
    
    # List to store features
    features2concat = []

    try:
        # Load oddball blinks and similarity
        blinks = pd.read_csv(os.path.join(eog_folder, 
                                        f'{base}_blinks.csv'))
        similarity = pd.read_csv(os.path.join(eog_folder, 
                                              f'{base}_similarity.csv'))
        waveforms = pd.read_csv(os.path.join(eog_folder,
                                             f'{base}_waveforms.csv'),
                                index_col=0)
            
        # Blink-blink intervals
        if blinks.shape[0] > 0:
            blinks_BB = np.hstack([[np.nan],
                                    np.diff(blinks['Peak'].values)
                                    ])
        else:
            blinks_BB = np.empty(0)
        
        # amplitudes   
        amps = waveforms.iloc[:,np.argmin(np.abs(waveforms.columns.astype(float)))].values
        
        # Create feature entry for this channel
        feature_row = pd.DataFrame({
                        'Condition': condition,
                        'Subject': sub,
                        'Time': blinks['Peak'],
                        'Duration': blinks['Offset'] - blinks['Onset'],
                        'Rise': blinks['Peak'] - blinks['Onset'],
                        'Fall': blinks['Offset'] - blinks['Peak'],
                        'Amplitude': amps,
                        'BB': blinks_BB,
                        'Similarity': similarity['Similarity'],
                        })
        
    except (FileNotFoundError,KeyError,ValueError) as e:
        print(f"Error for {base}:\n{e}")
        feature_row = pd.DataFrame({
                                    'Condition': condition,
                                    'Subject': sub,
                                    'Time': np.nan,
                                    'Duration': np.nan,
                                    'Rise': np.nan,
                                    'Fall': np.nan,
                                    'Amplitude': np.nan,
                                    'BB': np.nan,
                                    'Similarity': np.nan,
                                    },
                                    index=[0])
        
    finally:
        feature_row['Condition'] = feature_row['Condition'].astype(str)
        feature_row['Subject'] = feature_row['Subject'].astype(str)
        
    features2concat.append(feature_row)
            
    
    # Save features
    df_features = pd.concat(features2concat)
    path_save = os.path.join(eog_folder, f'{base}_features.csv')
    df_features.to_csv(path_save, index=False)
    
    return path_save

# ----------------- Utils ---------------------

def waveform_similarity(waveforms,
                        pearson = False):
    W = waveforms.to_numpy()
    n_waveforms, n_times = W.shape
    
    # Compute the similarity matrix
    S = np.zeros((n_waveforms, n_waveforms))
    
    _W = W - np.nanmean(W, axis=1)[:, np.newaxis] if pearson else W.copy() 
        
    for i in range(n_waveforms):
        for j in range(n_waveforms):
            norm_i = np.sqrt(np.nansum(_W[i]*_W[i]))
            norm_j = np.sqrt(np.nansum(_W[j]*_W[j]))
            S[i,j] = np.nansum(_W[i]*_W[j]) / (norm_i * norm_j)
    
    return S


def _plot_colormapped_waveforms(waveforms, Savg, ax, 
                                cmap='cividis',
                                dark_or_light='dark',
                                invert_cmap=False,
                                alpha=0.25,
                                ):
    """
    Plot waveforms on a given axis with line color mapped to Savg using a colormap.

    Parameters:
    - waveforms: array of shape (Nwaveforms, Ntimepoints)
    - times: array of shape (Ntimepoints,)
    - Savg: array of shape (Nwaveforms,), values in [0, 1]
    - cmap: matplotlib colormap object
    - ax: matplotlib Axes object to plot into
    - label: optional title label for the plot
    """
    
    cmap = __cmap_or_cmap_from_color(cmap, 
                                     dark_or_light='blend:#000000,',
                                     reversed=invert_cmap)
    
    n_waveforms, n_timepoints = waveforms.shape
    waveforms_ms = waveforms.copy().T
    waveforms_ms.index = waveforms_ms.index.astype(float) * 1000  # Convert to ms
    
    # mpl version
    sim_as_color = [cmap(s) for s in np.array(Savg).squeeze()]
    waveforms_ms.plot(ax=ax,
                    legend=False,
                    color=sim_as_color,
                    alpha=alpha,)

    # Clean aesthetics
    #ax.set_title(label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Amplitude ($\mu V$)')


def threshold_survival_curve(xx, thresholds):
    '''Compute percentage of values in xx surviving each threshold.
       For a threshold t, out is P(xx > t).
    '''
    return [np.mean(xx > t) for t in thresholds]

@ignore_runtime_warning
def plotter_waveforms(waveforms, 
                      sim, 
                      blinks, 
                      cmap='cividis') -> plt.Figure:
    """
    Plot blink waveforms with similarity-based coloring and associated blink metrics.

    This function generates a multi-panel figure to visualize blink waveforms, their
    similarity, and associated metrics such as peak amplitudes, onset, and offset times.

    Parameters
    ----------
    waveforms : pd.DataFrame
        DataFrame containing blink waveforms. Rows represent individual blinks, and columns
        represent time points relative to the blink peak.
    sim : np.ndarray
        Array of similarity values for each blink waveform, used for coloring.
    blinks : pd.DataFrame
        DataFrame containing blink metrics with columns 'Amplitude', 'Onset', 'Offset', and 'Peak'.
    cmap : str or matplotlib colormap, optional
        Colormap to use for coloring waveforms based on similarity. Default is 'cividis'.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    # Determine the time vector
    t_vec = waveforms.columns.astype(float) * 1000  # Convert to milliseconds

    # Figure layout with GridSpec
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 3, width_ratios=[3, 1, 1], height_ratios=[3, 1])

    ax_waveforms = fig.add_subplot(gs[0, 0])  # Waveforms panel
    ax_amplitudes = fig.add_subplot(gs[0, 1])  # Peak amplitudes panel
    ax_onset_offset = fig.add_subplot(gs[1, 0])  # Onset/offset times panel
    ax_similarity = fig.add_subplot(gs[1, 1])  # Similarity survival curve panel

    # Plot waveforms with similarity-based coloring
    if waveforms.shape[0]:
        avg_waveform = np.nanmean(waveforms, axis=0)
        _plot_colormapped_waveforms(waveforms, sim, ax_waveforms, cmap=cmap)
        ax_waveforms.plot(t_vec, avg_waveform, color='black', linewidth=2)
        ax_waveforms.set_title("Blink Waveforms")
        ax_waveforms.set_ylabel(r'Amplitude [$\mu$V]')
        ax_waveforms.set_xlabel(r'Time [ms]')
    else:
        ax_waveforms.annotate('No data available',
                              xy=(0.5, 0.5), xycoords='axes fraction',
                              ha='center', va='center',
                              fontsize=12, color='red')

    # Plot peak amplitude distribution
    if 'Amplitude' in blinks.columns:
        peak_amplitudes = blinks['Amplitude'].values
        ax_amplitudes.boxplot(peak_amplitudes, 
                              vert=True, 
                              patch_artist=True, 
                              boxprops=dict(facecolor='lightgray'))
        ax_amplitudes.set_title("Peak Amplitudes")
        ax_amplitudes.set_ylabel(r"Amplitude [$\mu$V]")
        ax_amplitudes.set_ylim(ax_waveforms.get_ylim())
    else:
        ax_amplitudes.annotate('No data available',
                               xy=(0.5, 0.5), xycoords='axes fraction',
                               ha='center', va='center',
                               fontsize=12, color='red')

    # Plot onset and offset times
    if {'Onset', 'Offset', 'Peak'}.issubset(blinks.columns):
        onset_times = (blinks['Onset'].values - blinks['Peak'].values) * 1000
        offset_times = (blinks['Offset'].values - blinks['Peak'].values) * 1000
        ax_onset_offset.boxplot([onset_times, offset_times], vert=False, patch_artist=True,
                                labels=["Onset", "Offset"], boxprops=dict(facecolor='lightblue'))
        ax_onset_offset.set_title("Onset and Offset Times")
        ax_onset_offset.set_xlabel(r"Time [ms]")
    else:
        ax_onset_offset.annotate('No data available',
                                 xy=(0.5, 0.5), xycoords='axes fraction',
                                 ha='center', va='center',
                                 fontsize=12, color='red')

    # Plot similarity survival curve
    thresholds = np.linspace(0, 1, 256)
    if sim.size > 0:
        surviving_thresholds = threshold_survival_curve(sim, thresholds)
        ax_similarity.plot(thresholds, surviving_thresholds, color='k', linewidth=1.5)
        ax_similarity.set_title("Similarity Survival Curve")
        ax_similarity.set_xlabel("Similarity Threshold")
        ax_similarity.set_ylabel("Proportion Surviving")
    else:
        ax_similarity.annotate('No data available',
                                xy=(0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center',
                                fontsize=12, color='red')

    plt.tight_layout()
    return fig

