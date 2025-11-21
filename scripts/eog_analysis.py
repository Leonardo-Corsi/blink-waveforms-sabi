import os
from collections import namedtuple  # use for easy to read parametrization
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from dotenv import load_dotenv
from IPython.display import display  # noqa: F401
from joblib import Parallel, delayed
from mne import set_log_level
from tqdm import tqdm

from eogtools.blink_extraction import (
    blinks_from_eog,
    extract_blink_waveforms,
    extract_features_from_waveforms,
)
from utils.events import get_events_from_record

set_log_level('ERROR') # Set MNE log level to ERROR to avoid excessive output

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.spines.right': False,
    'axes.spines.top': False
})
plt.switch_backend('TkAgg') # to use interactive plotting, o.w. uses inline 
pd.options.display.float_format = '{:,.4g}'.format

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

# Prepare output directories
print(f"Working on output folder: {results_dir}")
os.makedirs(results_dir, exist_ok=True)
eog_dir = os.path.join(results_dir, 'eog')
os.makedirs(eog_dir, exist_ok=True)

print(f"Output folders {results_dir},{eog_dir} ready.")

lookfor = os.path.join(data_dir, 'sub-*','eog','sub-*.edf')
globbed = glob(lookfor) 

# Find blinks
Parallel(n_jobs=opt.n_jobs,verbose=5)(
        delayed(blinks_from_eog)(edfpath, eog_dir, 
                                 eog_channel = opt.eog_channel,
                                 blink_min_dur=opt.blink_min_dur,
                                 find_eog_kwargs=opt.find_eog_kwargs,
                                 )
        for edfpath in tqdm(globbed,
                            desc='Processing records',
                            unit='record',
                            )
    )

# Waveform extraction from blink
Parallel(n_jobs=opt.n_jobs,verbose=5)(
        delayed(extract_blink_waveforms)(os.path.splitext(os.path.basename(edfpath))[0], 
                                         eog_dir)
        for edfpath in tqdm(globbed,
                            desc='Getting waveforms from blinks information',
                            unit='record',
                            )
    )

# Feature extraction from waveforms
Parallel(n_jobs=opt.n_jobs,verbose=5)(
    delayed(extract_features_from_waveforms)(os.path.splitext(os.path.basename(edfpath))[0], 
                                         eog_dir)
        for edfpath in tqdm(globbed,
                             desc='Getting waveforms from blinks information',
                                unit='record',
                                )
    )

# stimulations
Parallel(n_jobs=opt.n_jobs,verbose=5)(
    delayed(get_events_from_record)(edfpath, eog_dir,regex=opt.stimulation_regex)
    for edfpath in tqdm(globbed,
                        desc='Getting events from stimulations',
                        unit='record',
                        )
)