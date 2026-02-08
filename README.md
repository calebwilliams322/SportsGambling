# SportsBetting

A deep learning system for NFL player prop predictions, built with PyTorch.

## Overview

Predicts raw player stat values (passing yards, rushing yards, receptions, etc.) using historical NFL data, then compares predictions against posted betting lines to find value. Built to be modular and extensible to other sports and bet types.

## Quick Start

```bash
git clone https://github.com/calebwilliams322/SportsGambling.git
cd SportsGambling
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download all NFL data (cached locally)
python scripts/download_data.py

# 2. Merge datasets and engineer features
python scripts/build_features.py

# 3. Train a model (one per stat type)
python scripts/train.py --stat passing_yards

# 4. Generate predictions
python scripts/predict.py --stat passing_yards
```

## Data Sources

All data sourced from [nfl_data_py](https://github.com/cooperdff/nfl_data_py) (nflverse ecosystem). No API keys required.

- Player weekly stats (1999-present)
- Game schedules, weather, Vegas lines
- Snap counts
- Next Gen Stats tracking data (2016+)
- Injury reports

## Project Structure

```
SportsBetting/
├── src/
│   ├── config.py          # Central configuration
│   ├── data/
│   │   ├── ingest.py      # Data fetching from nfl_data_py
│   │   ├── merge.py       # Dataset merging
│   │   └── features.py    # Feature engineering
│   ├── models/
│   │   └── player_props.py # PyTorch model definitions
│   └── training/
│       ├── dataset.py     # PyTorch Dataset & DataLoader
│       ├── trainer.py     # Training loop
│       └── evaluate.py    # Evaluation metrics
├── scripts/               # CLI entry points
├── data/                  # Raw & processed data (gitignored)
├── models/                # Saved weights (gitignored)
└── notebooks/             # EDA notebooks
```

## Status

Under active development — MVP targeting NFL player props.
