# Working Questions

Open questions to guide project planning and scoping.

## Sports & Use Cases

1. Which sports are we targeting first? (NFL, NBA, MLB, soccer, etc.)
   - **Answer: NFL first, but building the architecture to be sport-agnostic and extensible.**
2. What kind of bets are we focusing on?
   - ~~Moneyline (straight win/loss prediction)~~
   - ~~Spread (predicting margin of victory)~~
   - ~~Over/Under (total points/scores)~~
   - **Player props (individual player performance) — MVP focus**
   - ~~Live/in-game betting (real-time predictions)~~

## Data

3. Where are we sourcing data? (ESPN API, sports-reference, odds APIs, etc.)
4. How far back do we want historical data?
5. What types of features are we including?
   - Box score stats
   - Player tracking data
   - Weather
   - Injury reports
   - Line movement / odds history

## Model Architecture

6. What architecture(s) do we want to start with?
   - Feed-forward nets with embeddings for categorical features (teams/players)
   - Sequence models (LSTMs/Transformers over recent game history)
   - Graph neural networks (modeling player/team relationships)
   - Ensemble approaches combining multiple specialized models

## Strategy

7. Are we predicting outcomes, or specifically finding value against the betting lines?
   - **Answer: Predict raw stat values (e.g. passing yards, rushing yards, receptions) and compare against posted lines to find value.**
8. How are we thinking about bankroll management / Kelly criterion on the output side?

## Scope

9. What does the first iteration look like? (one sport, one bet type?)
   - **Answer: NFL player props MVP. Modular design so we can plug in new sports, bet types, and models over time.**
