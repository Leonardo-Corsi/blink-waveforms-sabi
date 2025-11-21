
import functools

#import pdb
import warnings
from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
from mne.filter import filter_data
from mne.preprocessing.eog import _get_eog_channel_index
from mne.utils import logger, verbose

from utils.rawtools import create_BAD_bykey, remove_BAD_segments

mne.set_log_level('ERROR')  # Set MNE log level to ERROR to avoid excessive output

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.spines.right': False,
    'axes.spines.top': False
})
plt.switch_backend('TkAgg') # to use interactive plotting, o.w. uses inline 


# Context manager for nan warnings, they are annoying
@contextmanager
def IgnoreRuntimeWarning():
    # Catch warnings in this block
    with warnings.catch_warnings():
        # Ignore all runtime warnings (including numpy "Mean of empty slice" etc.)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield  # Execute user code here

def ignore_runtime_warning(func):
    """
    Decorator that applies the NoNanWarning context manager
    around the decorated function.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with IgnoreRuntimeWarning():
            return func(*args, **kwargs)
    return wrapped

# find_eog_events by MNE with extra exposed parameters
@verbose
def find_eog_events(
    raw,
    # exposed extrema parameter
    extrema = None,
    # default kwargs as in mne.preprocessing.find_eog_events
    event_id=998,
    l_freq=1,
    h_freq=10,
    filter_length="10s",
    ch_name=None,
    tstart=0,
    reject_by_annotation=False,
    thresh=None, # changed: can be a callable!
    verbose=None,
):
    """Custom version of mne.preprocessing.find_eog_events to also 
    retrieve signal,extrema and events used for eog peaks.
    1. returns also the filtered signal used for peak finding
    2. exposes extrema parameter and allows to force it to -1 or 1 
    (default is None for mne's automatic choice)
    3. accepts a callable for thresh parameter

    .. note:: To control true-positive and true-negative detection rates, you
              may adjust the ``thresh`` parameter.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    event_id : int
        The index to assign to found events.
    l_freq : float
        Low cut-off frequency to apply to the EOG channel in Hz.
    h_freq : float
        High cut-off frequency to apply to the EOG channel in Hz.
    filter_length : str | int | None
        Number of taps to use for filtering.
    %(ch_name_eog)s
    tstart : float
        Start detection after tstart seconds.
    reject_by_annotation : bool
        Whether to omit data that is annotated as bad.
    thresh : float | None
        Threshold to trigger the detection of an EOG event. This controls the
        thresholding of the underlying peak-finding algorithm. Larger values
        mean that fewer peaks (i.e., fewer EOG events) will be detected.
        If ``None``, use the default of ``(max(eog) - min(eog)) / 4``,
        with ``eog`` being the filtered EOG signal.
    %(verbose)s

    Returns
    -------
    eog_events : array
        Events.
    
    filteog : np.ndarray
        Filtered signal used for peak finding.

    See Also
    --------
    mne.preprocessing.find_eog_events
    mne.preprocessing._peak_finder.peak_finder
    """
    
    # Getting EOG Channel
    eog_inds = _get_eog_channel_index(ch_name, raw)
    eog_names = np.array(raw.ch_names)[eog_inds]  # for logging
    logger.info(f"EOG channel index for this subject is: {eog_inds}")

    # Reject bad segments.
    reject_by_annotation = "omit" if reject_by_annotation else None
    eog, times = raw.get_data(
        picks=eog_inds, reject_by_annotation=reject_by_annotation, return_times=True
    )
    times = times * raw.info["sfreq"] + raw.first_samp

    eog_events,filteog = _find_eog_events(
        eog,
        extrema=extrema,
        ch_names=eog_names,
        event_id=event_id,
        l_freq=l_freq,
        h_freq=h_freq,
        sampling_rate=raw.info["sfreq"],
        first_samp=raw.first_samp,
        filter_length=filter_length,
        tstart=tstart,
        thresh=thresh, # reminder: can be callable
        verbose=verbose,
    )
    # #Map times to corresponding samples.
    # eog_events[:, 0] = np.round(times[eog_events[:, 0] - raw.first_samp]).astype(int)
    return times,eog_events, filteog # added filteog

@verbose
def _find_eog_events(
    eog,
    *,
    extrema,
    ch_names,
    event_id,
    l_freq,
    h_freq,
    sampling_rate,
    first_samp,
    filter_length="10s",
    tstart=0.0,
    thresh=None, # now can be a callable!
    verbose=None,
):
    """Find EOG events."""
    logger.info(
        "Filtering the data to remove DC offset to help "
        "distinguish blinks from saccades"
    )
    # filtering to remove dc offset so that we know which is blink and saccades
    # hardcode verbose=False to suppress filter param messages (since this
    # filter is not under user control)
    fmax = np.minimum(45, sampling_rate / 2.0 - 0.75)  # protect Nyquist
    filteog = np.array(
        [
            filter_data(
                x,
                sampling_rate,
                2,
                fmax,
                None,
                filter_length,
                0.5,
                0.5,
                phase="zero-double",
                fir_window="hann",
                fir_design="firwin2",
                verbose='ERROR',
            )
            for x in eog
        ]
    )
    temp = np.sqrt(np.sum(filteog**2, axis=1))
    indexmax = np.argmax(temp)
    if ch_names is not None:  # it can be None if called from ica_find_eog_events
        logger.info(f"Selecting channel {ch_names[indexmax]} for blink detection")

    # easier to detect peaks with filtering.
    filteog = filter_data(
        eog[indexmax],
        sampling_rate,
        l_freq,
        h_freq,
        None,
        filter_length,
        0.5,
        0.5,
        phase="zero-double",
        fir_window="hann",
        fir_design="firwin2",
        verbose='ERROR',
    )

    # detecting eog blinks and generating event file

    logger.info("Now detecting blinks and generating corresponding events")

    temp = filteog - np.mean(filteog)
    n_samples_start = int(sampling_rate * tstart)
    # added log and modified extrema handling
    
    # handling of thresh parameter
    if isinstance(thresh, float):
        _thresh = thresh
    elif thresh is None:
        _thresh = None
    elif isinstance(thresh, str) and thresh.startswith("q"):
        _thresh = np.quantile(filteog, float(thresh[1:]))
    else:
        raise ValueError("Invalid value for 'thresh': must be a float, "
                         "None, or 'q<float>' string for quantile")

    if extrema is None: 
        extrema = 1 if np.abs(np.max(temp)) > np.abs(np.min(temp)) else -1
        
    eog_events, _ = mne.preprocessing._peak_finder.peak_finder(filteog[n_samples_start:], 
                                                               _thresh, 
                                                               extrema=extrema)
    
    logger.info(f"""_find_eog_events started at sample {n_samples_start}, 
                looking for {'positive' if extrema==1 else 'negative'} peaks.""")
    
    eog_events += n_samples_start
    n_events = len(eog_events)
    logger.info(f"Number of EOG events detected: {n_events}")
    eog_events = np.array(
        [
            eog_events + first_samp,
            np.zeros(n_events, int),
            event_id * np.ones(n_events, int),
        ]
    ).T

    return eog_events,filteog # added filteog, removed eog_events

def compute_blink_durations( 
    eog_data: np.ndarray, 
    peak_samples: np.ndarray, 
    sfreq: float, 
    search_radius: float = 0.25,
    extrema: int = 1,
    savgol: dict = None,
    ) -> tuple[list[float], 
               list[float], 
               list[float], 
               list[int], 
               list[int]]: 
    _eog_data = np.squeeze(eog_data.copy())*extrema # allows to flip if needed
    if savgol is not None:
        if not isinstance(savgol, dict) and savgol:
            wl=search_radius/2*128
            wl = int(wl) if wl % 2 == 1 else int(wl) + 1  # Ensure window length is odd
            savgol = {
                    'window_length':wl,
                    'polyorder':2
                     }
        print(f"Filtering with savgol: ",savgol)
        _eog_data = scipy.signal.savgol_filter(_eog_data, **savgol)
    
    n_samples = _eog_data.size

    blink_onsets = []
    blink_offsets = []
    blink_durations = []
    blink_onset_samples = []
    blink_offset_samples = []

    search_window = int(round(search_radius * sfreq))
    peak_samples_sorted = np.sort(peak_samples)

    for peak_idx in peak_samples_sorted:
        # Left saddle
        left_start = max(0, peak_idx - search_window)
        seg_left = _eog_data[left_start:peak_idx]
        if seg_left.size == 0:
            blink_onsets.append(np.nan)
            blink_offsets.append(np.nan)
            blink_durations.append(np.nan)
            blink_onset_samples.append(np.nan)
            blink_offset_samples.append(np.nan)
            continue

        local_min_val = np.inf
        local_min_idx = np.nan
        for i in range(1, seg_left.size - 1):
            if seg_left[i] < seg_left[i - 1] and seg_left[i] < seg_left[i + 1] and seg_left[i] < local_min_val:
                local_min_val = seg_left[i]
                local_min_idx = i

        onset_idx = left_start if np.isnan(local_min_idx) else left_start + int(local_min_idx)

        # Right saddle
        right_end = min(n_samples, peak_idx + search_window)
        seg_right = _eog_data[peak_idx:right_end]
        if seg_right.size == 0:
            blink_onsets.append(onset_idx / sfreq)
            blink_offsets.append(np.nan)
            blink_durations.append(np.nan)
            blink_onset_samples.append(onset_idx)
            blink_offset_samples.append(np.nan)
            continue

        local_min_val = np.inf
        local_min_idx = np.nan
        for i in range(1, seg_right.size - 1):
            if seg_right[i] < seg_right[i - 1] and seg_right[i] < seg_right[i + 1] and seg_right[i] < local_min_val:
                local_min_val = seg_right[i]
                local_min_idx = i

        offset_idx = right_end - 1 if np.isnan(local_min_idx) else peak_idx + int(local_min_idx)

        if offset_idx < 0:
            offset_idx = 0
        if offset_idx > (n_samples - 1):
            offset_idx = n_samples - 1

        onset_time = onset_idx / sfreq
        offset_time = offset_idx / sfreq
        duration = offset_time - onset_time

        if onset_time > peak_idx / sfreq:
            raise RuntimeError(
                f"""Onset time {onset_time} is not before peak time 
                {peak_idx / sfreq} for peak at index {peak_idx}."""
            )
        if offset_time < peak_idx / sfreq:
            raise RuntimeError(
                f"""Offset time {offset_time} is not after peak time 
                {peak_idx / sfreq} for peak at index {peak_idx}."""
            )
        if onset_time > offset_time:
            raise RuntimeError(
                f"""Onset time {onset_time} is not before offset time 
                {offset_time} for peak at index {peak_idx}."""
            )
        blink_onsets.append(onset_time)
        blink_offsets.append(offset_time)
        blink_durations.append(duration)
        blink_onset_samples.append(onset_idx)
        blink_offset_samples.append(offset_idx)

    return (
        blink_onsets,
        blink_offsets,
        blink_durations,
        blink_onset_samples,
        blink_offset_samples
    )

def process_eog(
    edfpath: str,
    eog_channel: str | tuple[str, str] | list[str, str],
    extrema: int,
    blink_min_durations: float=0.1,
    find_eog_kwargs: dict[str, Any] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # ------------------- 0) Read data ----------------------------------------
    
    if find_eog_kwargs is None:
        find_eog_kwargs = {}
    raw_full = mne.io.read_raw_edf(edfpath, preload=True, verbose='ERROR')
    # Pick EOG channel data
    eogch = eog_channel if isinstance(eog_channel, list | tuple) else [eog_channel]
    picks = mne.pick_channels(raw_full.ch_names, eogch)
    if len(picks)==0:
        print(f"\nNo channels found matching {eog_channel} in {edfpath}")
        return (pd.DataFrame(), pd.DataFrame())

    raw_eog = raw_full.copy().pick(picks)
    ch_name = raw_full.ch_names[picks[0]]
    assert ch_name == eogch[0], f"Channel name mismatch: {ch_name} != {eogch[0]}"
        
    sfreq = raw_eog.info['sfreq']
    assert sfreq == 128, f"Unexpected sampling frequency: {sfreq} Hz"
    
    times_abs = raw_eog.times + raw_eog.first_time
    
    # Update find_eog_kwargs with the channel name
    _find_eog_kwargs = find_eog_kwargs.copy()
    _find_eog_kwargs["ch_name"] = ch_name
    
    # ------------------- 1) Removal of movement annotations ------------------
    bad_movement_annotations = create_BAD_bykey(raw_eog, 
                                                key='movimento',
                                                before_and_after=(10, 0))
    raw_eog_nomov = remove_BAD_segments(raw_eog.copy().set_annotations(bad_movement_annotations),
                                             interp='pchip')
    
    # ------------------- 4) EOG peak detection  ------------------------ 
    if np.isnan(raw_eog_nomov.get_data()).any():
        print(edfpath)
        print('nan')
        
    
    _, final_eog_events, final_eog_filtered = find_eog_events(raw_eog_nomov,
                                                              extrema=extrema,
                                                              **_find_eog_kwargs)
    
    blink_times_abs_all = final_eog_events[:, 0] / sfreq
    blink_indices_abs_all = final_eog_events[:, 0]
    blink_indices_rel_all = final_eog_events[:, 0] - raw_eog_nomov.first_samp
    blink_amplitudes_all = final_eog_filtered[blink_indices_rel_all]
    
    # ------------------- 5) Low-amplitude blink removal ----------------------
    if (isinstance(_find_eog_kwargs['thresh'],str) 
        and _find_eog_kwargs['thresh'].startswith("q")):
        _blink_min_amplitude = np.quantile(extrema*final_eog_filtered, 
                                            float(_find_eog_kwargs['thresh'][1:])
                                            )
    else:
        _blink_min_amplitude = _find_eog_kwargs['thresh']
    print(f'[{edfpath}] > min_blink_amplitude: {_blink_min_amplitude*1e6:.3f} µV')
    amplitude_mask = extrema*blink_amplitudes_all >= _blink_min_amplitude
    # safety measure
    ampunder1mv_mask = extrema*blink_amplitudes_all <= 1e-3 # protect against high noise
    amplitude_mask = amplitude_mask & ampunder1mv_mask
    blink_indices_abs_A = blink_indices_abs_all[amplitude_mask]
    blink_indices_rel_A = blink_indices_rel_all[amplitude_mask]
    blink_times_abs_A = blink_times_abs_all[amplitude_mask]
    blink_amplitudes_A = blink_amplitudes_all[amplitude_mask]
    
    # ------------------- 6) Onset/offset detection ---------------------------
    (onsets_abs_A, 
    offsets_abs_A, 
    durations_A, 
    onsets_idx_abs_A, 
    offsets_idx_abs_A) = compute_blink_durations(final_eog_filtered, 
                                                blink_indices_abs_A, 
                                                sfreq, 
                                                search_radius=0.5, 
                                                savgol = {'window_length': 63, 'polyorder': 2},
                                                extrema=extrema)
    
    # ------------------- 7) Remove invalid durations -------------------------
    valid_mask = ~(np.isnan(onsets_abs_A) 
                | np.isnan(offsets_abs_A) 
                | np.isnan(durations_A)
                )
    blink_indices_abs_AV = blink_indices_abs_A[valid_mask]
    blink_indices_rel_AV = blink_indices_rel_A[valid_mask]
    blink_times_abs_AV = blink_times_abs_A[valid_mask]
    blink_amplitudes_AV = blink_amplitudes_A[valid_mask]
    onsets_abs_AV = np.array(onsets_abs_A)[valid_mask]
    offsets_abs_AV = np.array(offsets_abs_A)[valid_mask]
    durations_AV = np.array(durations_A)[valid_mask]
    onsets_idx_abs_AV = np.array(onsets_idx_abs_A)[valid_mask].astype(int)
    offsets_idx_abs_AV = np.array(offsets_idx_abs_A)[valid_mask].astype(int)
    
    # ------------------- 8) Remove short-duration blinks ---------------------
    duration_mask = (offsets_abs_AV - onsets_abs_AV) >= blink_min_durations
    blink_indices_abs_AVD = blink_indices_abs_AV[duration_mask]
    blink_indices_rel_AVD = blink_indices_rel_AV[duration_mask]
    blink_times_abs_AVD = blink_times_abs_AV[duration_mask]
    blink_amplitudes_AVD = blink_amplitudes_AV[duration_mask]
    onsets_abs_AVD = onsets_abs_AV[duration_mask]
    offsets_abs_AVD = offsets_abs_AV[duration_mask]
    durations_AVD = durations_AV[duration_mask]
    onsets_idx_abs_AVD = onsets_idx_abs_AV[duration_mask]
    offsets_idx_abs_AVD = offsets_idx_abs_AV[duration_mask]
        
    # ------------------- 9) Build the DataFrame of final peaks ---------------
    # Columns: 'channel', 'blink_peak','blink_onset','blink_offset','blink_amplitude'
    blink_df = pd.DataFrame({
        'Channel': [ch_name] * len(blink_indices_abs_AVD),
        'Onset': onsets_abs_AVD,
        'Peak': blink_times_abs_AVD,
        'Offset': offsets_abs_AVD,
        'Onset_index': onsets_idx_abs_AVD,
        'Offset_index': offsets_idx_abs_AVD,
        'Peak_index': blink_indices_abs_AVD,
        'Amplitude': blink_amplitudes_AVD,
    })
    
    # ------------------- 10) Build the signals dataframe ---------------------
    # Index: times
    # Columns: 'Channel', 'raw_eog','guess_peak', 'nonoise','mne_filtered', 
    # 'final_peak','rej_amp_peak','rej_dur_peak'
    # The _peak columns are np.nan apart from the peak amplitude at the corresponding time.
    signals_df = pd.DataFrame({
        'Channel': [ch_name] * len(times_abs),
        'raw_eog': raw_eog_nomov.notch_filter(50,verbose='ERROR').get_data()[0],
        'mne_filtered': final_eog_filtered,
        'threshold': _blink_min_amplitude,
        'final_peak': np.nan,
        'rej_amp_peak': np.nan,
        'rej_dur_peak': np.nan
    }, index=times_abs)
    # Fill in the peak amplitudes
    signals_df.loc[blink_times_abs_all, 'final_peak'] = blink_amplitudes_all
    signals_df.loc[blink_times_abs_A, 'rej_amp_peak'] = blink_amplitudes_A
    signals_df.loc[blink_times_abs_AVD, 'rej_dur_peak'] = blink_amplitudes_AVD

    return (blink_df, signals_df)

def _plot_signals_df(signals_df: pd.DataFrame, 
                     ax: plt.Axes,
                     ) -> plt.Axes:
    """
    Plot the signals_df with custom styles for each column.

    Parameters
    ----------
    signals_df : pd.DataFrame
        The DataFrame containing signal data to plot. Index should be time.
    title : str
        The title of the plot.
    """
    plt.rcParams.update({'font.size': 8})
    # Define custom styles for each column
    styles = {
        'raw_eog': {'linewidth': 1, 'color': 'gray', 'alpha': 0.3, 'label': 'Raw EOG'},
        'final_peak': {'linestyle': '', 'marker': 'o', 'color': 'gray', 'label': 'Peak'},
        'threshold': {'color': 'orange', 'linestyle':'--','linewidth': 1.5, 
                      'alpha':0.5,'label': 'Min. Amp.'},
        'mne_filtered': {'color': 'green', 'label': 'Filtered','linewidth': 0.5},
        'rej_dur_peak': {'linestyle': '', 'marker': 'd','markersize':1.5, 
                         'color': 'black', 'label': 'Final Peak'}
    }

    # Plot each column with its corresponding style
    for column, style in styles.items():
        if column in signals_df:
            ax.plot(
                signals_df.index,
                signals_df[column],
                **style
            )

    # Add labels, legend, and grid
    ax.set_xlabel('Time (s)')
    ax.xaxis.set_major_locator(plt.MultipleLocator(30))
    ax.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
    ax.set_ylabel(r'Amplitude ($\mu V$)')
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.rcParams.update({'font.size': 12})
    return ax
     
def plotter_eog(
    blinks_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    basename: str = '',
    figsize: tuple[int, int] = (8, 6)
) -> plt.Figure:
    
    fig, ax = plt.subplots(nrows=1, ncols=1,
                             figsize=figsize)
    ax.set_title(f'Blink Detection: {basename}')

    # Plot each channel's signals
    put_legend = False

    if signals_df.shape[0]:
        _plot_signals_df(signals_df, ax)
        #ax.set_title(f'Channel {ch}')
        ax.set_xlim(signals_df.index[0]-5, signals_df.index[-1]+5)
        filt_lims = [signals_df['mne_filtered'].min(),
                     signals_df['mne_filtered'].max()]
        filt_lims[0] = min(0,filt_lims[0]*1.5)
        filt_lims[1] = max(0,filt_lims[1]*1.5)
        ax.set_ylim(filt_lims)
        if put_legend:
            ax.legend(ncols=10,
                        loc='upper center')
            
                    
    else:
        ax.annotate('No data available',
                    xy=(0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center',
                    fontsize=12, color='red')
    
    # Add informations about the number of blinks
    # at the beginning, after amplitude rejection, and after duration rejection
    if signals_df.shape[0]:
        n_blinks_all = len(signals_df['final_peak'].dropna())
        n_blinks_A = len(signals_df['rej_amp_peak'].dropna())
        n_blinks_AD = len(signals_df['rej_dur_peak'].dropna())
    else:
        n_blinks_all, n_blinks_A, n_blinks_AD = 0,0,0
    
    # Add channel name at the top with blink information 
    ax.text(0.05, 0.95, 
            ''.join([
                    f"\n#Blinks Ok/Tot: {n_blinks_AD}/{n_blinks_all}",
                    f"\nRej.Amp/Tot: {n_blinks_all-n_blinks_A}/{n_blinks_all}",
                    f"\nRej.Dur/ok.Amp: {n_blinks_A-n_blinks_AD}/{n_blinks_A}",
                    ]),
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))    
    plt.tight_layout()
    return fig

