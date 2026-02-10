# SportsBetting

A deep learning system for NFL player prop predictions, built with PyTorch and a full-stack web UI.

## Overview

Predicts raw player stat values (passing yards, rushing yards, receptions, etc.) using historical NFL data, then compares predictions against posted betting lines to find value. Built to be modular and extensible to other sports and bet types.

Includes a **web GUI** (React + FastAPI) for running the full pipeline — download data, build features, train models, and generate predictions — all from the browser.

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

The Predictions page automatically pulls live player prop O/U lines from **DraftKings** (via [The Odds API](https://the-odds-api.com/)) and displays the **Edge** (prediction minus line) and a **Confidence Score** for each player.

### Confidence Score

Each prediction with a sportsbook line gets a confidence score (1-99) that combines:

1. **Edge z-score (65%)** — How many of the player's own standard deviations the edge represents. A 20-yard edge on a player with 10-yard std (z=2) is far more meaningful than on a player with 30-yard std (z=0.67).
2. **Market alignment (35%)** — Converts the DraftKings American odds for our direction (over or under) into implied probability. If the market agrees with our lean, confidence gets a boost.

| Score | Label | Meaning |
|---|---|---|
| 70+ | High | Large edge relative to player variance, market may agree |
| 40-69 | Moderate | Meaningful edge but more uncertainty |
| <40 | Low | Small edge or high player variance |

## Web GUI

The project includes a full-stack web interface for running the entire pipeline without touching the command line.

### Starting the GUI

```bash
# Terminal 1: Start the backend
cd backend
python server.py

# Terminal 2: Start the frontend
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

### Pages

**Dashboard** — Overview of your data: total rows, seasons, player count, feature count. Shows pipeline status and top 2025 passers.

**Data Pipeline** — Step-by-step pipeline execution:
1. **Download NFL Data** — Pull weekly stats, schedules, snap counts, NGS, and injuries (2016-2025)
2. **Build Features** — Merge all datasets and engineer 200+ rolling/trend/opponent features
3. **Train Model** — Links to Training page
4. **Predict** — Links to Predictions page

Each step shows real-time streaming logs and status indicators.

**Data Explorer** — Browse raw and processed datasets in table format. Click any dataset (weekly, schedules, snap counts, NGS, injuries, merged, features) to preview its contents.

**Training** — Train and manage neural network models:
- **Model Cards Grid** — Shows all trained models with test MAE, improvement % over baseline, epochs trained, and feature count
- **Train/Retrain** — Select a stat type (passing_yards, rushing_yards, etc.), optionally configure hyperparameters (learning rate, hidden layer sizes, dropout, patience), and click Train
- **Advanced Settings** — Collapsible panel for hyperparameter tuning
- **Live Logs** — Watch training progress in real-time (epoch-by-epoch loss, early stopping, evaluation results)

**Predictions** — Auto-populated predictions vs. live sportsbook lines:
- **Select Model** — Pick a trained stat model (shows MAE for each)
- **Select Game** — Auto-fill from schedule (loads playoff/postseason games with spread, total, venue) or adjust spread/total manually
- **Auto-populate** — Instantly generates predictions for all starters on both teams (filtered to active roster via 5-week recency window)
- **Sportsbook Comparison** — Automatically fetches live DraftKings player prop O/U lines via The Odds API
- **Edge + Confidence** — Shows the edge (prediction minus line) and a confidence score combining edge z-score, DK odds, and player variance
- Results split into home/away team tables, sorted by predicted value

### Backend API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Pipeline step statuses and available data files |
| `/api/download` | POST | Trigger data download (background thread) |
| `/api/build-features` | POST | Trigger feature engineering (background thread) |
| `/api/logs/{step}` | GET | SSE stream for real-time logs (download, build_features, train) |
| `/api/data-preview/{dataset}` | GET | Preview first N rows of any dataset |
| `/api/data-stats` | GET | Summary statistics for the features dataset |
| `/api/models` | GET | List of trained models with metrics |
| `/api/train` | POST | Train/retrain a model (accepts stat + optional hyperparams) |
| `/api/players?stat=X&team=Y` | GET | Eligible players for a stat, filtered by position and team |
| `/api/schedule-games` | GET | Playoff/Super Bowl games for auto-filling game context |
| `/api/predict` | POST | Single-player prediction with game context overrides |
| `/api/predict-batch` | POST | Batch predictions for all starters on both teams |
| `/api/props?stat=X&home_team=Y&away_team=Z` | GET | Live sportsbook player prop lines from The Odds API |

### Error Handling

The backend returns clear error messages for invalid states:
- Train without features built: "Features not built yet. Run the data pipeline first."
- Train unknown stat: "Unknown stat. Available: passing_yards, rushing_yards, ..."
- Predict without trained model: "No trained model for passing_yards. Train it first."
- Predict unknown player: "Player 'X' not found. Did you mean: ..." (fuzzy matching)
- Predict wrong position: "Player X (WR) not eligible for passing_yards predictions (requires QB)"

These are shown in styled error banners in the UI.

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

- **Architecture:** Feed-forward neural network (128 -> 64 -> 32 -> 1) with ReLU activations and dropout
- **Loss function:** Huber loss (more robust to outlier games than MSE)
- **Optimizer:** Adam
- **Train/Val/Test split:** Chronological cutoff (train: 2016-2025 wk14, val: 2025 wk15-18, test: 2025 wk19-21) — no temporal leakage
- **Features normalized:** StandardScaler fit on training data only
- **Targets normalized:** Mean/std normalization fit on training data, predictions denormalized back to original scale
- **Early stopping:** Stops training when validation loss stops improving (patience: 10 epochs)
- **One model per stat:** `passing_yards`, `passing_tds`, `rushing_yards`, `carries`, `receptions`, `receiving_yards`, `receiving_tds`
- **Position filtering:** Only trains on relevant positions per stat (e.g. QBs for passing_yards, WR/TE/RB for receiving_yards)

### 5. Prediction (`scripts/predict.py`)

Loads a trained model, builds input features for the requested players using their most recent data, and outputs predicted stat values.

For the web UI, predictions are game-specific: the player's rolling stats are combined with game context (opponent, spread, total, venue) and fed through the model. The prediction is denormalized using the target mean/std saved in the model's metadata.

## Quick Start

### Option A: Web GUI (recommended)

```bash
# Clone and install Python dependencies
git clone https://github.com/calebwilliams322/SportsGambling.git
cd SportsGambling
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

# Start the backend (Terminal 1)
# Set ODDS_API_KEY to enable live sportsbook line comparison (optional, free at https://the-odds-api.com)
ODDS_API_KEY=your_key_here python backend/server.py

# Start the frontend (Terminal 2)
cd frontend
npm run dev
```

Open http://localhost:5173 and use the UI to:
1. Download data (Data Pipeline page)
2. Build features (Data Pipeline page)
3. Train models (Training page)
4. Generate predictions (Predictions page)

The repo includes pre-trained models for passing_yards, rushing_yards, and receiving_yards, plus all the raw and processed data, so you can skip straight to the Training and Predictions pages.

### Option B: Command Line

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
├── backend/
│   └── server.py                  # FastAPI backend (API endpoints, log streaming)
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # React app (Dashboard, Pipeline, Explorer, Training, Predictions)
│   │   ├── App.css                # Component styles
│   │   ├── index.css              # Design variables (colors, spacing)
│   │   └── main.jsx               # React entry point
│   ├── package.json               # React 19, Vite
│   └── vite.config.js             # Vite dev server config
├── src/
│   ├── config.py                  # Seasons, targets, feature lists, hyperparameters
│   ├── data/
│   │   ├── ingest.py              # Pulls data from nfl_data_py
│   │   ├── merge.py               # Joins all datasets together
│   │   └── features.py            # Rolling averages, opponent stats, etc.
│   ├── models/
│   │   └── player_props.py        # PyTorch neural net definition
│   └── training/
│       ├── dataset.py             # PyTorch Dataset, DataLoader, train/val/test splits
│       ├── trainer.py             # Training loop with early stopping
│       └── evaluate.py            # MAE, RMSE, baseline comparison
├── scripts/
│   ├── download_data.py           # Entry point: pull & cache raw data
│   ├── build_features.py          # Entry point: merge + feature engineering
│   ├── train.py                   # Entry point: train a model
│   └── predict.py                 # Entry point: generate predictions
├── data/
│   ├── raw/                       # Cached parquet files (weekly, schedules, NGS, etc.)
│   └── processed/                 # Merged & engineered features
├── models/                        # Saved model weights (.pt) + metadata (.json)
└── notebooks/                     # Exploration notebooks
```

## Pre-trained Models

The repo ships with 3 pre-trained models so you can generate predictions immediately:

| Stat | Test MAE | Improvement vs Baseline |
|---|---|---|
| Passing Yards | 66.5 | 26.6% |
| Rushing Yards | 8.2 | 55.4% |
| Receiving Yards | 18.7 | 20.1% |

Models are saved as `models/{stat}_best.pt` (weights) and `models/{stat}_meta.json` (feature columns, normalization stats, metrics, config).

## Tech Stack

- **PyTorch** — neural network framework
- **FastAPI** — backend API server
- **React + Vite** — frontend web UI
- **nfl_data_py** — NFL data (no API keys needed)
- **The Odds API** — live sportsbook player prop lines (free tier, optional)
- **pandas** — data manipulation
- **scikit-learn** — preprocessing (StandardScaler)
- **pyarrow** — parquet file I/O

## Status

Under active development — MVP targeting NFL player props.
