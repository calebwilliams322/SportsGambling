# TODO — Remaining Steps

## Completed
- [x] Step 1: Project scaffolding (directories, .gitignore, requirements.txt, config)
- [x] Step 2: Data ingestion (nfl_data_py + PBP fallback for 2025)
- [x] Step 3: Data merging (weekly + schedules + snaps + NGS + injuries)
- [x] Step 4: Feature engineering (rolling 3/5/8, trends, opponent defense, NGS, temporal)
- [x] GUI: Vite + FastAPI frontend for steps 1-4

## Up Next
- [ ] Step 5: Train a model — run `python3 scripts/train.py --stat passing_yards` and verify loss goes down, MAE beats baseline
- [ ] Step 6: Train all stat models — `python3 scripts/train.py --all` (passing_yards, passing_tds, rushing_yards, carries, receptions, receiving_yards, receiving_tds)
- [ ] Step 7: Evaluate — review MAE/RMSE on test set (2025 season), compare to naive baseline
- [ ] Step 8: Predict Super Bowl — `python3 scripts/predict.py --stat passing_yards --players "Patrick Mahomes,..."`
- [ ] Step 9: Compare predictions to posted lines — find value bets
- [ ] Step 10: Wire up training + prediction into the GUI

## Future Improvements
- [ ] Save scaler with model weights (currently re-fits from training data)
- [ ] Position filtering (only predict passing_yards for QBs, receptions for WR/TE/RB, etc.)
- [ ] Hyperparameter tuning (learning rate, hidden sizes, dropout, batch size)
- [ ] LSTM/Transformer sequence model (feed last N games as a sequence)
- [ ] Odds API integration (auto-pull current prop lines)
- [ ] EDA notebook for data exploration
- [ ] Expand to other sports (NBA, MLB)
- [ ] Bankroll management / Kelly criterion
- [ ] 2025 injury data (not yet available on nflverse)
- [ ] Performance optimization (pandas fragmentation warnings in features.py)
