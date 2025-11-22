"""
update_ongoing_games.py
Incrementally updates game_history with completed games AND stats since last update
PLUS verifies prediction accuracy
"""

import psycopg2
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from io import StringIO
import time
from datetime import datetime, timedelta


def get_last_update_date(season='2025-26'):
    """Get the most recent game date in database for current season"""
    conn = psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )

    query = """
        SELECT MAX(game_date) as last_date
        FROM game_history
        WHERE season = %s
    """

    result = pd.read_sql(query, conn, params=(season,))
    conn.close()

    last_date = result['last_date'].iloc[0]

    if pd.isna(last_date):
        print(f"No games found for {season}. Starting from Oct 22, 2024")
        return datetime(2024, 10, 22)

    print(f"Last game in database: {last_date}")
    return pd.to_datetime(last_date)


def get_teams_list():
    """Get list of all teams from database"""
    conn = psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )

    teams = pd.read_sql("SELECT team_name, team_abbr FROM teams", conn)
    conn.close()

    return teams


def scrape_team_games_since_date(team_name, team_abbr, start_date, season='2025-26'):
    """
    Scrape a team's games since start_date (both schedule and stats)
    Returns merged DataFrame with complete game data
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Determine URL year (season ending year)
    url_year = '2026' if season == '2025-26' else season.split('-')[1]

    # 1. Get team schedule (basic game info)
    games_url = f"https://www.basketball-reference.com/teams/{team_abbr}/{url_year}_games.html"

    try:
        response = requests.get(games_url, headers=headers)
        time.sleep(2)

        games = pd.read_html(StringIO(response.text),
                             match="Regular Season Table")[0]
        games = games[games['G'] != 'G'].reset_index(drop=True)

        # Filter for completed games after start_date
        games = games[games['LOG'].notna()].reset_index(drop=True)

        if len(games) == 0:
            return pd.DataFrame()

        # Parse dates and filter
        games['Date'] = pd.to_datetime(games['Date'])
        games = games[games['Date'] > start_date].reset_index(drop=True)

        if len(games) == 0:
            return pd.DataFrame()

        print(f"  {team_name}: Found {len(games)} new games")

        # 2. Get team stats (detailed stats from game log)
        game_log_url = f"https://www.basketball-reference.com/teams/{team_abbr}/{url_year}/gamelog/"
        stats_response = requests.get(game_log_url, headers=headers)
        time.sleep(2)

        game_stats = pd.read_html(
            StringIO(stats_response.text), match=f"{season} Regular Season Table")[0]
        game_stats.columns = game_stats.columns.droplevel()
        game_stats['Rk'] = pd.to_numeric(game_stats['Rk'], errors='coerce')
        game_stats = game_stats[game_stats['Rk'].notna()
                                ].reset_index(drop=True)

        # Keep only first 30 columns (team stats, not opponent stats)
        team_stats_only = game_stats.iloc[:, :30]

        # Parse dates and filter
        team_stats_only['Date'] = pd.to_datetime(team_stats_only['Date'])
        team_stats_only = team_stats_only[team_stats_only['Date'] > start_date].reset_index(
            drop=True)

        # 3. Merge games and stats by date
        merged = games.merge(
            team_stats_only[['Date', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
                             '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%',
                             'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']],
            on='Date',
            how='left'
        )

        # Add team identifiers
        merged['Team_Name'] = team_name
        merged['Team_Abbr'] = team_abbr
        merged['Season'] = season

        # Determine home/away
        soup = BeautifulSoup(response.text, 'html.parser')
        at_symbols = []
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            for cell in cells:
                if cell.get('data-stat') == 'game_location':
                    at_symbols.append(1 if cell.get_text(
                        strip=True) != '@' else 0)

        if len(at_symbols) >= len(merged):
            merged['Home'] = at_symbols[:len(merged)]
        else:
            merged['Home'] = None

        return merged

    except Exception as e:
        print(f"  Error scraping {team_name}: {e}")
        return pd.DataFrame()


def transform_to_game_history_format(df):
    """Transform merged data to game_history table format"""

    if df.empty:
        return df

    # Rename columns to match database
    column_mapping = {
        'Team_Name': 'team_name',
        'Team_Abbr': 'team_abbr',
        'Season': 'season',
        'G': 'game_number',
        'Date': 'game_date',
        'Start (ET)': 'start_time',
        'Opponent': 'opponent',
        'Home': 'home_status',
        'Tm': 'team_score',
        'Opp': 'opp_score',
        'W': 'win_number',
        'L': 'loss_number',
        'Streak': 'streak',
        'Attend.': 'attend',
        'LOG': 'game_length',
        'Notes': 'notes',
        'FG': 'fg_made',
        'FGA': 'fg_att',
        'FG%': 'fg_percent',
        '3P': 'threep_made',
        '3PA': 'threep_att',
        '3P%': 'threep_percent',
        '2P': 'twop_made',
        '2PA': 'twop_att',
        '2P%': 'twop_percent',
        'FT': 'ft_made',
        'FTA': 'ft_att',
        'FT%': 'ft_percent',
        'ORB': 'orb',
        'DRB': 'drb',
        'TRB': 'total_rb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        'TOV': 'tov',
        'PF': 'pf'
    }

    # Select and rename columns
    cols_to_keep = [col for col in column_mapping.keys() if col in df.columns]
    df_clean = df[cols_to_keep].copy()
    df_clean = df_clean.rename(columns=column_mapping)

    return df_clean


def insert_games_to_database(games_df):
    """Insert new games with stats into database"""

    if games_df.empty:
        print("No games to insert")
        return 0

    conn = psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )
    cur = conn.cursor()

    # Check for existing games
    cur.execute("""
        SELECT team_name, game_date, opponent
        FROM game_history
        WHERE season = '2025-26'
    """)

    existing = {f"{row[0]}_{row[1]}_{row[2]}" for row in cur.fetchall()}

    # Filter out duplicates
    games_df['unique_key'] = (
        games_df['team_name'] + '_' +
        games_df['game_date'].astype(str) + '_' +
        games_df['opponent']
    )

    new_games = games_df[~games_df['unique_key'].isin(existing)]

    if len(new_games) == 0:
        print("All games already in database")
        cur.close()
        conn.close()
        return 0

    print(f"Inserting {len(new_games)} new game records with stats...")

    inserted = 0
    for _, row in new_games.iterrows():
        try:
            cur.execute("""
                INSERT INTO game_history (
                    team_name, team_abbr, season, game_number, game_date, start_time,
                    opponent, home_status, team_score, opp_score,
                    win_number, loss_number, streak, attend, game_length, notes,
                    fg_made, fg_att, fg_percent,
                    threep_made, threep_att, threep_percent,
                    twop_made, twop_att, twop_percent,
                    ft_made, ft_att, ft_percent,
                    orb, drb, total_rb, ast, stl, blk, tov, pf
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
            """, (
                row['team_name'],
                row['team_abbr'],
                row['season'],
                int(row['game_number']) if pd.notna(
                    row.get('game_number')) else None,
                row['game_date'],
                row.get('start_time'),
                row['opponent'],
                int(row['home_status']) if pd.notna(
                    row.get('home_status')) else 0,
                int(row['team_score']) if pd.notna(
                    row.get('team_score')) else None,
                int(row['opp_score']) if pd.notna(
                    row.get('opp_score')) else None,
                int(row['win_number']) if pd.notna(
                    row.get('win_number')) else None,
                int(row['loss_number']) if pd.notna(
                    row.get('loss_number')) else None,
                row.get('streak'),
                int(row['attend']) if pd.notna(row.get('attend')) else None,
                row.get('game_length'),
                row.get('notes'),
                float(row['fg_made']) if pd.notna(
                    row.get('fg_made')) else None,
                float(row['fg_att']) if pd.notna(row.get('fg_att')) else None,
                float(row['fg_percent']) if pd.notna(
                    row.get('fg_percent')) else None,
                float(row['threep_made']) if pd.notna(
                    row.get('threep_made')) else None,
                float(row['threep_att']) if pd.notna(
                    row.get('threep_att')) else None,
                float(row['threep_percent']) if pd.notna(
                    row.get('threep_percent')) else None,
                float(row['twop_made']) if pd.notna(
                    row.get('twop_made')) else None,
                float(row['twop_att']) if pd.notna(
                    row.get('twop_att')) else None,
                float(row['twop_percent']) if pd.notna(
                    row.get('twop_percent')) else None,
                float(row['ft_made']) if pd.notna(
                    row.get('ft_made')) else None,
                float(row['ft_att']) if pd.notna(row.get('ft_att')) else None,
                float(row['ft_percent']) if pd.notna(
                    row.get('ft_percent')) else None,
                float(row['orb']) if pd.notna(row.get('orb')) else None,
                float(row['drb']) if pd.notna(row.get('drb')) else None,
                float(row['total_rb']) if pd.notna(
                    row.get('total_rb')) else None,
                float(row['ast']) if pd.notna(row.get('ast')) else None,
                float(row['stl']) if pd.notna(row.get('stl')) else None,
                float(row['blk']) if pd.notna(row.get('blk')) else None,
                float(row['tov']) if pd.notna(row.get('tov')) else None,
                float(row['pf']) if pd.notna(row.get('pf')) else None
            ))
            inserted += 1
        except Exception as e:
            print(
                f"Error inserting game for {row['team_name']} on {row['game_date']}: {e}")
            continue

    conn.commit()
    cur.close()
    conn.close()

    return inserted


def update_scheduled_games_results():
    """
    Update scheduled_games table with actual scores from game_history
    """
    conn = psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )
    cur = conn.cursor()

    print("\nUpdating scheduled game results...")

    # Find scheduled games that have completed but scores not yet recorded
    query = """
        SELECT 
            sg.scheduled_game_id,
            sg.game_date,
            sg.home_team_id,
            sg.away_team_id,
            ht.team_name as home_team,
            at.team_name as away_team
        FROM scheduled_games sg
        JOIN teams ht ON sg.home_team_id = ht.team_id
        JOIN teams at ON sg.away_team_id = at.team_id
        WHERE sg.completed = FALSE
          AND sg.game_date < CURRENT_DATE
    """

    incomplete_games = pd.read_sql(query, conn)

    if incomplete_games.empty:
        print("  No scheduled games to update")
        cur.close()
        conn.close()
        return 0

    print(f"  Found {len(incomplete_games)} scheduled games to check")

    updated = 0
    for _, game in incomplete_games.iterrows():
        # Get actual results from game_history
        cur.execute("""
            SELECT team_score, opp_score
            FROM game_history
            WHERE team_name = %s
              AND opponent = %s
              AND game_date = %s
              AND season = '2025-26'
        """, (game['home_team'], game['away_team'], game['game_date']))

        result = cur.fetchone()

        if result:
            home_score, away_score = result

            # Update scheduled_games
            cur.execute("""
                UPDATE scheduled_games
                SET home_score = %s,
                    away_score = %s,
                    completed = TRUE
                WHERE scheduled_game_id = %s
            """, (int(home_score), int(away_score), game['scheduled_game_id']))

            updated += 1
            print(
                f"    {game['away_team']} @ {game['home_team']}: {away_score}-{home_score}")

    conn.commit()
    cur.close()
    conn.close()

    print(f"  ✓ Updated {updated} scheduled games with results")
    return updated


def update_prediction_results():
    """
    Update predictions table with actual game results
    Marks predictions as correct/incorrect
    """
    conn = psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )
    cur = conn.cursor()

    print("\nUpdating prediction results...")

    # Get predictions that don't have actual results yet
    query = """
        SELECT 
            p.prediction_id,
            p.scheduled_game_id,
            p.home_team_id,
            p.away_team_id,
            p.game_date,
            p.predicted_home_win,
            p.predicted_winner_id,
            sg.home_score,
            sg.away_score,
            sg.completed
        FROM predictions p
        JOIN scheduled_games sg ON p.scheduled_game_id = sg.scheduled_game_id
        WHERE p.actual_home_win IS NULL
          AND sg.completed = TRUE
          AND sg.home_score IS NOT NULL
          AND sg.away_score IS NOT NULL
    """

    predictions_to_update = pd.read_sql(query, conn)

    if predictions_to_update.empty:
        print("  No predictions to update")
        cur.close()
        conn.close()
        return 0

    print(f"  Found {len(predictions_to_update)} predictions to verify")

    updated = 0
    for _, pred in predictions_to_update.iterrows():
        # Determine actual winner
        actual_home_win = pred['home_score'] > pred['away_score']
        actual_winner_id = pred['home_team_id'] if actual_home_win else pred['away_team_id']

        # Check if prediction was correct
        was_correct = (pred['predicted_winner_id'] == actual_winner_id)

        # Update prediction
        cur.execute("""
            UPDATE predictions
            SET actual_home_win = %s,
                was_correct = %s
            WHERE prediction_id = %s
        """, (actual_home_win, was_correct, pred['prediction_id']))

        updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"  ✓ Updated {updated} prediction results")
    return updated


def show_prediction_accuracy():
    """
    Display overall prediction accuracy statistics
    """
    conn = psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )

    query = """
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN was_correct = TRUE THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(
                SUM(CASE WHEN was_correct = TRUE THEN 1 ELSE 0 END)::numeric / 
                COUNT(*)::numeric * 100, 
                2
            ) as accuracy_percentage
        FROM predictions
        WHERE actual_home_win IS NOT NULL
    """

    stats = pd.read_sql(query, conn)
    conn.close()

    if not stats.empty and stats['total_predictions'].iloc[0] > 0:
        print("\n" + "="*60)
        print("PREDICTION ACCURACY SUMMARY")
        print("="*60)
        print(
            f"Total Predictions Verified: {int(stats['total_predictions'].iloc[0])}")
        print(
            f"Correct Predictions: {int(stats['correct_predictions'].iloc[0])}")
        print(f"Accuracy: {stats['accuracy_percentage'].iloc[0]}%")
        print("="*60)


def update_ongoing_games():
    """
    Main function: Update database with games AND stats since last update
    PLUS update prediction results
    """
    print("="*60)
    print("NBA Game History Update (With Stats)")
    print("="*60)

    # 1. Get last update date
    last_date = get_last_update_date('2025-26')

    # 2. Get all teams
    teams = get_teams_list()
    print(f"\nLoaded {len(teams)} teams")

    # 3. Scrape each team's new games
    print(f"\nScraping games since {last_date.date()}...")
    all_new_games = []

    for _, team in teams.iterrows():
        try:
            team_games = scrape_team_games_since_date(
                team['team_name'],
                team['team_abbr'],
                last_date,
                season='2025-26'
            )

            if not team_games.empty:
                all_new_games.append(team_games)

        except Exception as e:
            print(f"  Error with {team['team_name']}: {e}")
            continue

    if not all_new_games:
        print("\nNo new completed games found")
    else:
        # 4. Combine all teams' data
        combined = pd.concat(all_new_games, ignore_index=True)
        print(f"\nTotal new games found: {len(combined)}")

        # 5. Transform to database format
        games_formatted = transform_to_game_history_format(combined)

        # 6. Insert into database
        print("\nInserting into database...")
        inserted = insert_games_to_database(games_formatted)

        print("\n" + "="*60)
        print(f"✓ Game History Update Complete: {inserted} new records added")
        print("="*60)

        # 7. Show summary
        conn = psycopg2.connect(
            dbname="nba_stats",
            user="kaedenlee",
            host="localhost"
        )

        summary = pd.read_sql("""
            SELECT 
                season,
                COUNT(*) as total_games,
                MIN(game_date) as first_game,
                MAX(game_date) as last_game
            FROM game_history
            WHERE season = '2025-26'
            GROUP BY season
        """, conn)

        conn.close()

        if not summary.empty:
            print("\nCurrent Season Summary:")
            print(summary.to_string(index=False))

    # 8. Update scheduled_games with results
    update_scheduled_games_results()

    # 9. Update prediction results
    update_prediction_results()

    # 10. Show prediction accuracy
    show_prediction_accuracy()

    print("\n" + "="*60)
    print("✓ Complete Update Finished!")
    print("="*60)


if __name__ == "__main__":
    update_ongoing_games()
