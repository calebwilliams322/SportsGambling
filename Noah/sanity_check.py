"""Sanity check: show recent game logs for each recommended bet player."""

import pandas as pd
from data.fetch import fetch_weekly_stats

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

weekly = fetch_weekly_stats()

players = {
    "passing_yards": [
        ("Drake Maye", ["season", "week", "season_type", "recent_team", "opponent_team",
                        "attempts", "completions", "passing_yards", "passing_tds",
                        "interceptions", "passing_epa"]),
    ],
    "rushing_yards": [
        ("Kenneth Walker III", ["season", "week", "season_type", "recent_team", "opponent_team",
                                "carries", "rushing_yards", "rushing_tds", "rushing_epa"]),
    ],
    "receiving_yards": [
        ("Jaxon Smith-Njigba", ["season", "week", "season_type", "recent_team", "opponent_team",
                                 "targets", "receptions", "receiving_yards", "receiving_tds",
                                 "target_share", "receiving_epa"]),
        ("Rashid Shaheed", ["season", "week", "season_type", "recent_team", "opponent_team",
                            "targets", "receptions", "receiving_yards", "receiving_tds",
                            "target_share", "receiving_epa"]),
        ("Kayshon Boutte", ["season", "week", "season_type", "recent_team", "opponent_team",
                            "targets", "receptions", "receiving_yards", "receiving_tds",
                            "target_share", "receiving_epa"]),
    ],
}

for prop_type, player_list in players.items():
    for player_name, cols in player_list:
        print(f"\n{'='*90}")
        print(f"  {player_name} — {prop_type.replace('_', ' ').upper()}")
        print(f"{'='*90}")

        data = weekly[weekly["player_display_name"] == player_name].copy()
        if data.empty:
            # Try partial match
            matches = weekly[weekly["player_display_name"].str.contains(player_name.split()[0], case=False, na=False)]
            unique = matches["player_display_name"].unique()
            print(f"  NOT FOUND. Possible matches: {list(unique[:10])}")
            continue

        data = data.sort_values(["season", "week"])

        # Show career summary
        seasons = data["season"].unique()
        print(f"  Seasons in data: {list(seasons)}")
        print(f"  Total games: {len(data)}")
        print(f"  Teams: {list(data['recent_team'].unique())}")

        # Show last 10 games
        recent = data[cols].tail(10)
        print(f"\n  Last {len(recent)} games:")
        print(recent.to_string(index=False))

        # Stats from last 5 games (what the model uses)
        last5 = data.tail(5)
        target_col = prop_type.replace("_yards", "_yards")
        if prop_type in last5.columns:
            avg = last5[prop_type].mean()
            median = last5[prop_type].median()
            low = last5[prop_type].min()
            high = last5[prop_type].max()
            print(f"\n  Last 5 games {prop_type}: avg={avg:.1f}, median={median:.1f}, range=[{low:.0f}-{high:.0f}]")

        # Full season averages for context
        for s in sorted(seasons)[-2:]:
            szn = data[data["season"] == s]
            if prop_type in szn.columns:
                avg = szn[prop_type].mean()
                games = len(szn)
                print(f"  {s} season avg: {avg:.1f} {prop_type.replace('_', ' ')} over {games} games")
