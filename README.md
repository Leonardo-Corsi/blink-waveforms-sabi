# Strength and Timing Failure Modes in Auditory Stimulus-Locked Blink Synchronization Across Levels of Consciousness

This branch contains the analysis pipeline for the manuscript:

> **Strength and Timing Failure Modes in Auditory Stimulus-Locked Blink Synchronization Across Levels of Consciousness**
>
> Leonardo Corsi, Alfonso Magliacano, Piergiuseppe Liuzzi, Calogero Maria Oddo, Anna Estraneo, Andrea Mannini

Paper-specific development is maintained in the [`blink-probability-modulation-sabi`](https://github.com/leonardocorsi/blink-waveforms-sabi/tree/blink-probability-modulation-sabi) branch of this repository. The [`main`](https://github.com/leonardocorsi/blink-waveforms-sabi) branch remains the reference analysis for the earlier blink-waveform study.

The analysis reuses the common EOG blink-extraction pipeline and adds stimulus-locked timing analyses designed to separate the strength of blink modulation from its temporal alignment to the auditory stimulation cycle.

## Overview

### Pipeline stages

1. **Participant metadata** - Loads demographics and clinical group information.

   -> `scripts/demographics_data.py`, `data/demographics.csv`

2. **Blink and stimulus extraction** - Processes vertical EOG recordings, detects blinks, extracts blink waveforms and features, and recovers auditory stimulus onsets.

   -> `scripts/eog_analysis.py`, `src/eogtools/`, `src/utils/events.py`

3. **Circular timing analysis** - Tests raw blink delays for a preferred position within the repeated stimulus cycle using subject-level Rayleigh statistics and group-level summaries.

   -> `scripts/model_MBP.py`

4. **Mean blink proportion analysis** - Computes peri-stimulus blink counts and mean blink proportion (MBP), with resting data retained as a negative-control condition where appropriate.

   -> `scripts/model_MBP.py`

5. **Model-based MBP analysis** - Fits the literature Huber model for comparison and a low-dimensional periodic ramp model based on Neuwirth (2001) to estimate subject-level modulation strength and temporal delay.

   -> `scripts/model_MBP.py`

6. **Group statistics and figures** - Evaluates model fit, summarizes fitted parameters across HC, eMCS, and pDoC groups, and exports manuscript tables and figures.

   -> `scripts/model_MBP.py`, `results_MBP/`

Generated outputs include cached subject-level MBP/count data, Rayleigh summaries, model parameters, fit-quality measures, group statistics, and manuscript figures.

## Repository structure

```text
blink-waveforms-sabi/
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── configs/
│   └── opt.yml
├── data/
│   ├── demographics.csv
│   └── sub-*/eog/sub-*_task-*.edf
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
│   └── model_MBP.py
├── results*/
│   ├── eog/
│   └── STIM/
└── results_MBP/
    ├── cache/
    ├── figures/
    └── tables/
```

Raw EDF files are expected under the BIDS-like `data/sub-*/eog/` layout when preprocessing is rerun, but they do not need to be versioned with the repository.

## Installation

Dependencies are declared in `pyproject.toml` and require Python >= 3.12.7.

```text
git clone --branch blink-probability-modulation-sabi --single-branch https://github.com/leonardocorsi/blink-waveforms-sabi.git
cd blink-waveforms-sabi
```

Using `pip`:

```text
python -m venv .venv
source .venv/bin/activate
pip install .
```

Using `conda`:

```text
conda create -n blink-sabi python=3.12.7
conda activate blink-sabi
pip install .
```

## Configuration

Local environment variables can be defined in an untracked `.env` file for preprocessing:

```text
DATA_DIR=./data
RESULTS_DIR=./results
CONFIG_PATH=./configs/opt.yml
PYTHONPATH=src
```

`configs/opt.yml` controls EOG processing parameters, the stimulation-event matching expression, plotting palettes, and parallel job count.

By default, `scripts/model_MBP.py` reads blink and stimulus files from `./results`. If preprocessing outputs are stored elsewhere, set the `MBP_INPUT_RESULTS_DIR` environment variable to that directory before running the MBP analysis.

## Running the analysis

From the repository root:

```text
python scripts/demographics_data.py
python scripts/eog_analysis.py
python scripts/model_MBP.py
```

The EOG preprocessing step writes blink and stimulus files under `RESULTS_DIR`. The paper-specific analysis writes its outputs to `results_MBP/`, organized into `cache/`, `figures/`, and `tables/`.

The committed `results_MBP/` directory provides the analysis products associated with the current branch snapshot. Re-running the scripts may overwrite files with the same names.

## Citation

If you use the analysis in this branch, please cite the accompanying manuscript and this repository. Citation metadata are provided in `CITATION.cff`.

## License

Distributed under the MIT License - see `LICENSE`.
