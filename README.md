# SportsBetting

A deep learning system for NFL player prop predictions, built with PyTorch.

## Overview

Predicts raw player stat values (passing yards, rushing yards, receptions, etc.) using historical NFL data, then compares predictions against posted betting lines to find value. Built to be modular and extensible to other sports and bet types.

## How It Works

### The Core Idea

We train **one neural network per stat type** (e.g. one model for passing yards, one for rushing yards, one for receptions, etc.). Each model is trained on **every player's historical games** — not one model per player.

The model never sees a player's name or identity directly. Instead, it learns general patterns like:
- "A player averaging 280 passing yards over their last 3 games, with a 98% snap share, against a defense allowing 260 passing yards per game, in a dome, as a slight favorite... tends to throw for ~290 yards."

These patterns apply across all players, which is key because any single player only has ~17 games per season — not nearly enough data to train a neural net on their own.

### Training

During training, the model sees thousands of rows, where each row is one player's one game:

| Player's Rolling Avg (3g) | Snap % | Opp Def Avg | Home? | Spread | ... | **Actual Yards** |
|---|---|---|---|---|---|---|
| 285 | 0.98 | 260 | 1 | -3.5 | ... | **312** |
| 195 | 0.95 | 230 | 0 | +7.0 | ... | **178** |
| 62 | 0.88 | 105 | 1 | -2.5 | ... | **71** |

The model learns the relationship between the input features (everything we knew *before* the game) and the output (what actually happened).

### Prediction

When we want to predict a specific player for an upcoming game, we build **one input row** for that player using their most recent data:

For example, predicting Patrick Mahomes' passing yards in the Super Bowl:

| Feature | Value | Source |
|---|---|---|
| `passing_yards_avg_3` | 285 | His last 3 games |
| `passing_yards_avg_5` | 271 | His last 5 games |
| `passing_yards_avg_8` | 268 | His last 8 games |
| `passing_yards_trend` | +17 | Short-term vs long-term avg |
| `passing_yards_std_3` | 42 | How consistent he's been |
| `snap_pct_avg_3` | 0.98 | His snap share recently |
| `opp_def_passing_yards_avg` | 245 | What opponent allows to QBs |
| `is_home` | 0 | Away game |
| `spread_line` | -1.5 | Vegas spread |
| `total_line` | 49.5 | Expected total points |
| `is_dome` | 1 | Indoor game |
| `temp` | 72 | Dome temperature |
| `injury_status_code` | 4 | Healthy |
| `ngs_passing_avg_time_to_throw_avg_3` | 2.7 | NGS tracking metric |
| ... | ... | ... |

We feed this row into the trained model, and it outputs a single number: **predicted passing yards**.

We then do the same for every other player we care about — each gets their own unique input row built from *their* rolling stats, *their* matchup context, etc.

### Finding Value

Once we have predictions, we compare them to the sportsbook's posted lines:
- Model predicts Mahomes throws for **290 yards**
- The book has the over/under at **275.5 yards**
- Our model says **over** — that's a potential value bet

## Data Pipeline

### 1. Data Ingestion (`scripts/download_data.py`)

Pulls 7 datasets from [nfl_data_py](https://github.com/cooperdff/nfl_data_py) and caches them locally as parquet files:

| Dataset | What It Contains | Used For |
|---|---|---|
| Weekly Stats | Every player's box score stats per game | Core training data (features + targets) |
| Schedules | Game info, weather, Vegas lines | Game context features |
| Snap Counts | Offensive snap percentages per player | Usage/opportunity features |
| NGS Passing | Tracking metrics (time to throw, aggressiveness) | Advanced QB features |
| NGS Rushing | Tracking metrics (efficiency, time to line of scrimmage) | Advanced RB features |
| NGS Receiving | Tracking metrics (separation, cushion, YAC) | Advanced WR/TE features |
| Injuries | Weekly injury reports and practice status | Availability features |

### 2. Data Merging (`scripts/build_features.py` — Step 1)

Joins all 7 datasets into a single master DataFrame. Each row = one player, one game, with all contextual data attached.

Merge keys:
- Weekly stats (base table) keyed on `player_id`, `season`, `week`
- Schedules joined on `season`, `week`, `team`
- Snap counts joined on `player_name`, `team`, `season`, `week`
- NGS data joined on `player_display_name`, `team_abbr`, `season`, `week`
- Injuries joined on `player_name`, `team`, `season`, `week`

### 3. Feature Engineering (`scripts/build_features.py` — Step 2)

All features are **strictly backward-looking** — we only use data available *before* the game being predicted. This prevents data leakage.

**Rolling Player Averages** (3, 5, and 8 game windows):
- Every stat column gets a rolling mean and standard deviation
- Example: `passing_yards_avg_3` = average passing yards over last 3 games

**Trend Features:**
- Difference between short-term (3-game) and long-term (8-game) averages
- Positive = player is trending up, negative = trending down

**Season-to-Date Averages:**
- Expanding mean of each stat within the current season

**Opponent Defensive Features:**
- Rolling average of stats allowed by the upcoming opponent's defense to each position group
- Example: `opp_def_passing_yards_avg` = how many passing yards this defense allows to QBs per game

**Snap Share:**
- Rolling average of offensive snap percentage (opportunity proxy)

**Game Context:**
- Home/away, Vegas spread, over/under total, dome/outdoor, temperature, wind

**Injury Encoding:**
- Practice status (DNP / Limited / Full) encoded numerically
- Game status (Out / Doubtful / Questionable / Probable) encoded numerically

**Temporal:**
- Week number, post-bye indicator, weeks since last game

**Next Gen Stats (rolling):**
- Passing: time to throw, aggressiveness, completion % above expectation
- Rushing: efficiency, time to line of scrimmage
- Receiving: separation, cushion, YAC above expectation

### 4. Model Training (`scripts/train.py`)

- **Architecture:** Feed-forward neural network (128 → 64 → 32 → 1) with ReLU activations and dropout
- **Loss function:** Huber loss (more robust to outlier games than MSE)
- **Optimizer:** Adam
- **Train/Val/Test split:** Walk-forward by season (train: 2016-2022, val: 2023, test: 2024)
- **Features normalized:** StandardScaler fit on training data only
- **Early stopping:** Stops training when validation loss stops improving (patience: 10 epochs)
- **One model per stat:** `passing_yards`, `passing_tds`, `rushing_yards`, `carries`, `receptions`, `receiving_yards`, `receiving_tds`

### 5. Prediction (`scripts/predict.py`)

Loads a trained model, builds input features for the requested players using their most recent data, and outputs predicted stat values.

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
# 1. Download all NFL data (cached locally, takes a few minutes first time)
python scripts/download_data.py

# 2. Merge datasets and engineer features
python scripts/build_features.py

# 3. Train a model for a specific stat
python scripts/train.py --stat passing_yards

# 4. Train models for ALL stats
python scripts/train.py --all

# 5. Generate predictions for all players
python scripts/predict.py --stat passing_yards

# 6. Predict for specific players
python scripts/predict.py --stat passing_yards --players "Patrick Mahomes,Jalen Hurts"
```

## Project Structure

```
SportsBetting/
├── src/
│   ├── config.py              # Seasons, targets, feature lists, hyperparameters
│   ├── data/
│   │   ├── ingest.py          # Pulls data from nfl_data_py
│   │   ├── merge.py           # Joins all datasets together
│   │   └── features.py        # Rolling averages, opponent stats, etc.
│   ├── models/
│   │   └── player_props.py    # PyTorch neural net definition
│   └── training/
│       ├── dataset.py         # PyTorch Dataset, DataLoader, train/val/test splits
│       ├── trainer.py         # Training loop with early stopping
│       └── evaluate.py        # MAE, RMSE, baseline comparison
├── scripts/
│   ├── download_data.py       # Entry point: pull & cache raw data
│   ├── build_features.py      # Entry point: merge + feature engineering
│   ├── train.py               # Entry point: train a model
│   └── predict.py             # Entry point: generate predictions
├── data/
│   ├── raw/                   # Cached parquet files (gitignored)
│   └── processed/             # Merged & engineered features (gitignored)
├── models/                    # Saved model weights + metadata (gitignored)
└── notebooks/                 # Exploration notebooks
```

## Tech Stack

- **PyTorch** — neural network framework
- **nfl_data_py** — NFL data (no API keys needed)
- **pandas** — data manipulation
- **scikit-learn** — preprocessing (StandardScaler)
- **pyarrow** — parquet file I/O

## Status

Under active development — MVP targeting NFL player props.
