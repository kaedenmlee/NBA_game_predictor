"""
update_ongoing_games.py
Incrementally updates game_history with completed games AND stats since last update
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
        FROM test
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

        # 3. Merge games and stats by game number
        games['Game_Num'] = range(1, len(games) + 1)
        team_stats_only['Game_Num'] = range(1, len(team_stats_only) + 1)

        # Match by date instead (more reliable for partial season)
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
        FROM test
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
                INSERT INTO test (
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


def update_ongoing_games():
    """
    Main function: Update database with games AND stats since last update
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
        return

    # 4. Combine all teams' data
    combined = pd.concat(all_new_games, ignore_index=True)
    print(f"\nTotal new games found: {len(combined)}")
    # combined.to_csv('new_games.csv', index=False)

    # 5. Transform to database format
    games_formatted = transform_to_game_history_format(combined)
    # games_formatted.to_csv('formatted_games.csv', index=False)

    # 6. Insert into database
    print("\nInserting into database...")
    inserted = insert_games_to_database(games_formatted)

    print("\n" + "="*60)
    print(f"✓ Update Complete: {inserted} new records added")
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
        FROM test
        WHERE season = '2025-26'
        GROUP BY season
    """, conn)

    conn.close()

    if not summary.empty:
        print("\nCurrent Season Summary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    update_ongoing_games()
