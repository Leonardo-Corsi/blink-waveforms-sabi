from itertools import groupby

import mne
import numpy as np
import pandas as pd
from mne._fiff.pick import _picks_to_idx


def remove_BAD_segments(
    raw: mne.io.Raw, 
    clip_range: tuple[float, float] = (-np.inf, np.inf), 
    interp: str | None = None,
    match: str = 'BAD', 
    picks: str | list | np.ndarray | tuple | None = None
) -> mne.io.Raw:
    """Remove segments annotated as BAD_* (or matching a custom pattern) from raw data.
    
    Parameters:
    ----------
    raw : mne.io.Raw
        The raw data object containing the data and annotations.
    clip_range : tuple, optional
        The range to clip the data to. Default is (-np.inf, np.inf).
    interp : str | None, optional
        The interpolation method to use for filling in the removed segments. 
        If None, no interpolation is performed. Default is None.
    match : str, optional
        The string pattern to match in annotation descriptions. Default is 'BAD'.
    picks : str | list | np.ndarray | tuple | None, optional
        The channels to process. If None, all channels are processed. Default is None.
    
    Returns:
    -------
    mne.io.Raw
        The cleaned raw object with noisy segments removed and optionally interpolated.
    """

    # Resolve picks to indices
    picks = _picks_to_idx(raw.info, picks, "all", exclude=[], allow_empty=False)
    data = raw.copy().get_data(picks=picks)
    sfreq = raw.info['sfreq']
    data_cleaned = data.copy()
    n_samples = raw.n_times
    
    # Construct a boolean vector of the indexes affected by matching annotations
    bad_mask = np.zeros(n_samples, dtype=bool)
    for annot in raw.annotations:
        if match.lower() in annot['description'].lower():
            start_sample = int(annot['onset'] * sfreq)
            end_sample = start_sample + int(annot['duration'] * sfreq)
            bad_mask[start_sample:end_sample] = True
            
    data_cleaned[:, bad_mask] = np.nan
    
    # Optionally interpolate
    if interp is not None:
        for i in range(data_cleaned.shape[0]):  # Interpolate each channel independently
            if np.isnan(data_cleaned[i,0]):
                data_cleaned[i,0] = 0
            if np.isnan(data_cleaned[i,-1]):
                data_cleaned[i,-1] = 0
            data_cleaned[i, :] = pd.Series(data_cleaned[i, :]).interpolate(
                method=interp, fill_value='extrapolate'
            ).to_numpy()
    
    data_cleaned = np.clip(data_cleaned, *clip_range)
    
    raw_ret = raw.copy()
    raw_ret._data[picks, :] = data_cleaned  # Only update selected picks
    return raw_ret

def create_BAD_byamp(
    raw: mne.io.Raw, 
    threshold: float, 
    window_s: int = 1
    ) -> mne.Annotations:
    """
    Create annotations for segments of EEG data with amplitudes exceeding a specified threshold.
    This function identifies segments in the raw EEG data where the absolute amplitude 
    exceeds the given threshold. It extends these segments by a specified window size 
    (in seconds) on both sides and generates annotations for these "BAD_amplitude" segments.
    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data to analyze.
    threshold : float
        The amplitude threshold for marking segments as "BAD_amplitude".
    window_s : int, optional
        The window size (in seconds) to extend the "BAD_amplitude" segments on both sides.
        Default is 1 second.
    Returns:
    --------
    mne.Annotations
        Annotations object containing the "BAD_amplitude" segments with their onset times, 
        durations, and descriptions.
    Notes:
    ------
    - The function uses the sampling frequency (`sfreq`) from the raw data to calculate 
      the number of samples corresponding to the window size.
    - The annotations are created using the measurement date (`meas_date`) from the raw data.
    """                 
    
    bad_amplitude = np.abs(raw.get_data().copy()) >= threshold
    bad_amplitude = np.any(bad_amplitude, axis=0)
    
    # Extend annotations by window_s/2 at each extremity
    bad_amplitude_extended = np.zeros(len(bad_amplitude), dtype=bool)
    for i in range(len(bad_amplitude)):
        if bad_amplitude[i]:
            start = max(0,int(i - window_s * raw.info['sfreq'] // 2))
            end = min(int(i + window_s * raw.info['sfreq'] // 2), len(bad_amplitude))
            bad_amplitude_extended[start:end] = [True] * (end - start)
    
    # Create annotations for True ranges, using groupby
    bad_amplitude_annotations = []
    for k, g in groupby(enumerate(bad_amplitude_extended), key=lambda x: x[1]):
        if k:
            group = list(g)
            start = group[0][0]
            end = group[-1][0]
            bad_amplitude_annotations.append({
                'onset': start / raw.info['sfreq'],
                'duration': (end - start) / raw.info['sfreq'],
                'description': 'BAD_amplitude',
                'meas_date': raw.info['meas_date']
            })
        
    return mne.Annotations(
        onset=[annot['onset'] for annot in bad_amplitude_annotations],
        duration=[annot['duration'] for annot in bad_amplitude_annotations],
        description=[annot['description'] for annot in bad_amplitude_annotations],
        orig_time=raw.info['meas_date']
    )

def create_BAD_bykey(
    raw: mne.io.Raw, 
    key: str, 
    before_and_after: tuple[float, float] = (0.5, 0.5)
) -> mne.Annotations:
    """
    Create annotations for segments of EEG data matching a specific key, labeled as 'BAD_<key>'.
    
    Parameters:
    ----------
    raw : mne.io.Raw
        The raw EEG data object containing annotations.
    key : str
        The description of the annotation to be labeled as 'BAD_<key>'. Case insensitive.
    before_and_after : tuple[float, float], optional
        A tuple specifying the time (in seconds) to extend before and after the annotation onset and offset.
        Default is (0.5, 0.5).
    
    Returns:
    -------
    mne.Annotations
        A new Annotations object containing the adjusted 'BAD_<key>' annotations.
    """
    # Get the annotations from the raw object
    annotations = raw.annotations

    # Create a list to store the new annotations
    new_annotations = []

    # Iterate through the annotations
    for annot in annotations:
        if key.lower() in annot['description'].lower():
            # Calculate the start and end times for the BAD_<key> annotation
            start_time = annot['onset'] - before_and_after[0]
            end_time = annot['onset'] + annot['duration'] + before_and_after[1]

            # Add the new annotation to the list
            new_annotations.append({
                'onset': start_time,
                'duration': end_time - start_time,
                'description': f'BAD_{key}'
            })

    # Create an Annotations object from the new annotations
    bad_key_annotations = mne.Annotations(
        onset=[annot['onset'] for annot in new_annotations],
        duration=[annot['duration'] for annot in new_annotations],
        description=[annot['description'] for annot in new_annotations],
        orig_time=raw.info['meas_date']
    )

    return bad_key_annotations