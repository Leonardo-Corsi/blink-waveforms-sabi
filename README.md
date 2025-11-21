# Blink waveform analysis and feature extraction for severe Acquired Brain Injury

This repository contains the full analysis pipeline for the submission of:

> **Demographics-robust spontaneous eye blinking slowing in patients with severe acquired brain injury**  
> Alfonso Magliacano(1), Leonardo Corsi(1,2)°, Piergiuseppe Liuzzi(1), Calogero Maria Oddo(2), Andrea Mannini(1), Anna Estraneo(1), 2025.

(1) IRCCS Fondazione Don Carlo Gnocchi ONLUS, Florence, Italy

(2) Scuola Superiore Sant’Anna, Pisa, Italy

 ° corresponding author (lcorsi@dongnocchi.it or leonardo.corsi.app@gmail.com)

The code implements an automated extraction and characterization of blink-related EOG waveforms from recordings in edf format, followed by statistical comparison across clinical groups and task conditions and demographical correction.


## Overview

**Pipeline stages**

1. **Demographics** — Demographical information on patients and healthy subjects  

   → `scripts/demographics_data.py`
2. **EOG processing** — Processing of EOG, extraction of blinks 

   → `scripts/eog_analysis.py`, `src/eogtools/eog.py`, `src/eogtools/blink_extraction.py`, `src/utils/events.py`,  `src/utils/rawtools.py`

3. **Feature computation** — Derives amplitude, duration, rise/fall times, inter-blink intervals, similarity metrics.  

   → `scripts/eog_analysis.py`,`src/utils/features.py`
4. **Statistical analysis** — Summary and statistical testing of blink timing and waveform features and demographical correction

   → `scripts/demographics_data.py`

Outputs include per-subject blink features, summary plots, and group-level statistics reported in the manuscript

---------------------------------------------------------------------
Repository structure

```text
blink-pdoc/
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── .env
├── configs/
│   └── opt.yml
├── src/
│   ├── eogtools/
│   │   ├── eog.py
│   │   └── blink_extraction.py
│   └── utils/
│       ├── events.py
│       ├── features.py
│       ├── plotting.py
│       └── rawtools.py
├── scripts/
│   ├── demographics_data.py
│   ├── eog_analysis.py
│   └── feature_stats.py
└── data/
    └── sub-*/eog/sub-*_task-*.edf
```
---------------------------------------------------------------------
Installation

All dependencies are declared in pyproject.toml (Python >= 3.12). 
Quick setup:
```text
git clone https://github.com/leonardocorsi/blink-pdoc.git
cd blink-pdoc
```

if using pip:
```text
python -m venv .venv
source .venv/bin/activate
pip install .
```

or, if using conda:
```text
conda create -n blink-pdoc python=3.12.7
conda activate blink-pdoc
pip install .
```

---------------------------------------------------------------------
Configuration

Environment variables are defined in .env:
```text
DATA_DIR=./data
RESULTS_DIR=./results
CONFIG_PATH=./configs/opt.yml
PYTHONPATH=src
```
The main YAML configuration configs/opt.yml controls instead:
- processing parameters (EOG channel, thresholds, window length)
- palette definitions for plotting
- parallel job number

---------------------------------------------------------------------
Running the analysis

1. Place EDF files in data/sub-*/eog/ using the BIDS-like naming convention:
   sub-<id>_task-Resting_eog.edf
   sub-<id>_task-Oddball_eog.edf
2. Edit .env and configs/opt.yml as needed.
3. Execute:
   python scripts/demographics_data.py
   python scripts/eog_analysis.py
4. Results (CSVs, SVGs, HTML) will appear in the RESULTS_DIR folder.


---------------------------------------------------------------------
Citation

If you use this code, please cite the accompanying publication and this repository (see CITATION.cff).

---------------------------------------------------------------------
License

Distributed under the MIT License — see LICENSE.