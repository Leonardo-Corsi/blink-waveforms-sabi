import pandas as pd
import os


def get_events(edfpath,contains_regex=None,min_dur=None,max_dur=None,min_onset=None,max_onset=None):
    from mne.io import read_raw_edf
    # Load annotations from .edf file 
    annots = read_raw_edf(edfpath,
                          preload=False, verbose='ERROR'
                         ).annotations
    if len(annots) == 0:
        return pd.DataFrame(columns=['onset', 'duration', 'description','orig_time'])
    # Build a stable DataFrame from MNE Annotations attributes to avoid pandas length-mismatch
    try:
        onsets = list(annots.onset)
        durations = list(annots.duration)
        descriptions = list(annots.description)
        orig_attr = getattr(annots, 'orig_time', None)
        if orig_attr is None:
            orig_times = [None] * len(onsets)
        elif hasattr(orig_attr, '__len__') and not isinstance(orig_attr, str | bytes):
            orig_times = list(orig_attr)
            if len(orig_times) != len(onsets):
                orig_times = [orig_attr] * len(onsets)
        else:
            orig_times = [orig_attr] * len(onsets)
        annots_df = pd.DataFrame({
            'onset': onsets,
            'duration': durations,
            'description': descriptions,
            'orig_time': orig_times
        })
    except Exception:
        # Fallback: safe per-annotation normalization
        rows = []
        for i in range(len(annots)):
            a = annots[i]
            d = dict(a) if not isinstance(a, dict) else a.copy()
            for k, v in d.items():
                if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                    d[k] = list(v) if len(v) != 1 else (v[0] if len(v) == 1 else None)
            rows.append(d)
        annots_df = pd.DataFrame(rows).reset_index(drop=True)
    # Filtering
    def series_or_str_contains(sstr):
        return sstr.astype(str).str.contains(contains_regex, regex=True, na=False)
    if contains_regex is not None:
        annots_df = annots_df[series_or_str_contains(annots_df['description'])]
    if min_dur is not None:
        annots_df = annots_df[annots_df['duration'] >= min_dur]
    if max_dur is not None:
        annots_df = annots_df[annots_df['duration'] <= max_dur]
    if min_onset is not None:
        annots_df = annots_df[annots_df['onset'] >= min_onset]
    if max_onset is not None:
        annots_df = annots_df[annots_df['onset'] <= max_onset]
    
    return annots_df

def get_events_from_record(
    edfpath,
    eog_folder,
    regex
):
    base = os.path.splitext(os.path.basename(edfpath))[0]
    
    stims = get_events(edfpath,contains_regex=regex)
    os.makedirs(os.path.join(eog_folder, '..','STIM'), exist_ok=True)
    stims.to_csv(os.path.join(eog_folder,'..','STIM',
                           f'{base}_stims.csv'),
                 index=False)
