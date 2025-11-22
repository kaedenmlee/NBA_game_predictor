# calculate_rolling_stats.py
import psycopg2
import pandas as pd
from datetime import datetime


def calculate_rolling_stats():
    """
    Calculate rolling statistics for all teams from game_history
    """
    conn = psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )

    # Load game history
    query = """
        SELECT 
            gh.*,
            t.team_id
        FROM game_history gh
        JOIN teams t ON gh.team_name = t.team_name
        ORDER BY t.team_id, gh.game_date
    """
    df = pd.read_sql(query, conn)
    df['game_date'] = pd.to_datetime(df['game_date'])

    print(f"Loaded {len(df)} games from game_history")

    # Calculate rolling stats per team/season
    rolling_data = []

    for (team_id, season), group in df.groupby(['team_id', 'season']):
        group = group.sort_values('game_date')

        # Calculate 10-game rolling averages
        window = 10
        group['avg_points'] = group['team_score'].rolling(
            window, min_periods=1).mean()
        group['avg_fg_pct'] = group['fg_percent'].rolling(
            window, min_periods=1).mean()
        group['avg_3p_pct'] = group['threep_percent'].rolling(
            window, min_periods=1).mean()
        group['avg_ft_pct'] = group['ft_percent'].rolling(
            window, min_periods=1).mean()
        group['avg_rebounds'] = group['total_rb'].rolling(
            window, min_periods=1).mean()
        group['avg_assists'] = group['ast'].rolling(
            window, min_periods=1).mean()
        group['avg_steals'] = group['stl'].rolling(
            window, min_periods=1).mean()
        group['avg_blocks'] = group['blk'].rolling(
            window, min_periods=1).mean()
        group['avg_turnovers'] = group['tov'].rolling(
            window, min_periods=1).mean()

        # Calculate win percentages
        group['won'] = group['team_score'] > group['opp_score']
        home_games = group[group['home_status'] == 1]
        away_games = group[group['home_status'] == 0]

        home_win_pct = home_games['won'].mean() if len(home_games) > 0 else 0
        away_win_pct = away_games['won'].mean() if len(away_games) > 0 else 0

        # Last 5 games record
        last_5_wins = group['won'].tail(5).sum() if len(
            group) >= 5 else group['won'].sum()

        # Get most recent stats
        if len(group) > 0:
            latest = group.iloc[-1]

            rolling_data.append({
                'team_id': team_id,
                'season': season,
                'as_of_date': latest['game_date'],
                'games_played': len(group),
                'avg_points': round(latest['avg_points'], 2),
                'avg_fg_pct': round(latest['avg_fg_pct'], 3),
                'avg_3p_pct': round(latest['avg_3p_pct'], 3),
                'avg_ft_pct': round(latest['avg_ft_pct'], 3),
                'avg_rebounds': round(latest['avg_rebounds'], 2),
                'avg_assists': round(latest['avg_assists'], 2),
                'avg_steals': round(latest['avg_steals'], 2),
                'avg_blocks': round(latest['avg_blocks'], 2),
                'avg_turnovers': round(latest['avg_turnovers'], 2),
                'home_win_pct': round(home_win_pct, 3),
                'away_win_pct': round(away_win_pct, 3),
                'last_5_wins': int(last_5_wins)
            })

    print(f"Calculated rolling stats for {len(rolling_data)} team-seasons")

    # Insert into database
    cur = conn.cursor()

    for stat in rolling_data:
        cur.execute("""
            INSERT INTO team_rolling_stats 
            (team_id, season, as_of_date, games_played, 
             avg_points, avg_fg_pct, avg_3p_pct, avg_ft_pct,
             avg_rebounds, avg_assists, avg_steals, avg_blocks, avg_turnovers,
             home_win_pct, away_win_pct, last_5_wins)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, season, as_of_date) 
            DO UPDATE SET
                games_played = EXCLUDED.games_played,
                avg_points = EXCLUDED.avg_points,
                avg_fg_pct = EXCLUDED.avg_fg_pct,
                avg_3p_pct = EXCLUDED.avg_3p_pct,
                avg_ft_pct = EXCLUDED.avg_ft_pct,
                avg_rebounds = EXCLUDED.avg_rebounds,
                avg_assists = EXCLUDED.avg_assists,
                avg_steals = EXCLUDED.avg_steals,
                avg_blocks = EXCLUDED.avg_blocks,
                avg_turnovers = EXCLUDED.avg_turnovers,
                home_win_pct = EXCLUDED.home_win_pct,
                away_win_pct = EXCLUDED.away_win_pct,
                last_5_wins = EXCLUDED.last_5_wins
        """, (
            int(stat['team_id']),                    # Convert to int
            str(stat['season']),                     # Convert to str
            stat['as_of_date'].date(),               # Convert to date
            int(stat['games_played']),               # Convert to int
            float(stat['avg_points']),               # Convert to float
            float(stat['avg_fg_pct']),
            float(stat['avg_3p_pct']),
            float(stat['avg_ft_pct']),
            float(stat['avg_rebounds']),
            float(stat['avg_assists']),
            float(stat['avg_steals']),
            float(stat['avg_blocks']),
            float(stat['avg_turnovers']),
            float(stat['home_win_pct']),
            float(stat['away_win_pct']),
            int(stat['last_5_wins'])                 # Convert to int
        ))

    conn.commit()
    cur.close()
    conn.close()

    print("✓ Rolling stats loaded to database!")

    return rolling_data


if __name__ == "__main__":
    stats = calculate_rolling_stats()

    # Show sample
    print("\n=== Sample Rolling Stats ===")
    for stat in stats[:5]:
        print(f"{stat['team_id']}: {stat['season']} - Avg Points: {stat['avg_points']}, "
              f"Home Win%: {stat['home_win_pct']}, Games: {stat['games_played']}")
