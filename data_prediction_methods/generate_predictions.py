import pickle
import pandas as pd
import psycopg2
from datetime import datetime


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )


def load_model(file_path='models/nba_model.pkl'):
    """Load trained model"""
    with open(file_path, 'rb') as f:
        model = pickle.load(f)

    print(f"✓ Model loaded from {file_path}")
    return model


def get_games_for_date(date):
    """
    Get games for a specific date from either scheduled_games or game_history
    """
    conn = get_db_connection()

    # First try scheduled_games
    query_scheduled = """
        SELECT 
            sg.scheduled_game_id,
            sg.game_date,
            sg.game_time,
            sg.home_team_id,
            sg.away_team_id,
            ht.team_name as home_team,
            ht.team_abbr as home_abbr,
            at.team_name as away_team,
            at.team_abbr as away_abbr
        FROM scheduled_games sg
        JOIN teams ht ON sg.home_team_id = ht.team_id
        JOIN teams at ON sg.away_team_id = at.team_id
        WHERE sg.game_date = %s
        ORDER BY sg.game_time
    """

    games = pd.read_sql(query_scheduled, conn, params=(date,))

    # If no scheduled games, try getting from game_history
    if games.empty:
        query_history = """
            SELECT 
                NULL as scheduled_game_id,
                gh.game_date,
                gh.start_time as game_time,
                ht.team_id as home_team_id,
                at.team_id as away_team_id,
                gh.team_name as home_team,
                ht.team_abbr as home_abbr,
                gh.opponent as away_team,
                at.team_abbr as away_abbr
            FROM game_history gh
            JOIN teams ht ON gh.team_name = ht.team_name
            JOIN teams at ON gh.opponent = at.team_name
            WHERE gh.game_date = %s
              AND gh.home_status = 1
              AND gh.season = '2025-26'
            ORDER BY gh.start_time
        """

        games = pd.read_sql(query_history, conn, params=(date,))

    conn.close()

    print(f"\n✓ Found {len(games)} games for {date}")
    return games


def get_team_rolling_stats(team_id, season='2025-26'):
    """Get latest rolling stats for a team"""
    conn = get_db_connection()

    query = """
        SELECT *
        FROM team_rolling_stats
        WHERE team_id = %s
          AND season = %s
        ORDER BY as_of_date DESC
        LIMIT 1
    """

    stats = pd.read_sql(query, conn, params=(team_id, season))
    conn.close()

    if stats.empty:
        return None

    return stats.iloc[0]


def prepare_game_features(home_team_id, away_team_id, game_time, game_date):
    """
    Prepare features for prediction
    Returns dict with all features needed by model
    """
    # Get rolling stats for both teams
    home_stats = get_team_rolling_stats(home_team_id)
    away_stats = get_team_rolling_stats(away_team_id)

    if home_stats is None or away_stats is None:
        print(f"  ⚠ Missing stats for teams {home_team_id} or {away_team_id}")
        return None

    # Extract hour from game time (e.g., "7:00p" -> 19)
    try:
        hour_str = game_time.split(':')[0] if game_time else '19'
        hour = int(hour_str)
        if 'p' in game_time.lower() and hour != 12:
            hour += 12
    except:
        hour = 19  # Default to 7pm

    # Get day of week
    day_code = pd.to_datetime(game_date).dayofweek

    # Create feature dict (matching training features)
    features = {
        'home_status': 1,  # Always 1 for home team
        'opp_code': away_team_id,  # Opponent code
        'team_code': home_team_id,  # Team code
        'hour': hour,
        'day_code': day_code,
        'rolling_avg_points': home_stats['avg_points'],
        'rolling_avg_fg_pct': home_stats['avg_fg_pct'],
        'rolling_avg_3p_pct': home_stats['avg_3p_pct'],
        'rolling_avg_ft_pct': home_stats['avg_ft_pct'],
        'rolling_avg_rebounds': home_stats['avg_rebounds'],
        'rolling_avg_assists': home_stats['avg_assists'],
        'rolling_avg_steals': home_stats['avg_steals'],
        'rolling_avg_blocks': home_stats['avg_blocks'],
        'rolling_avg_turnovers': home_stats['avg_turnovers'],
        'home_win_pct': home_stats['home_win_pct'],
        'away_win_pct': away_stats['away_win_pct'],
        'last_5_wins': home_stats['last_5_wins']
    }

    return features


def make_predictions(model, games_df):
    """
    Make predictions for all scheduled games
    """
    predictions = []

    for _, game in games_df.iterrows():
        print(f"\nPredicting: {game['away_team']} @ {game['home_team']}")

        # Prepare features
        features = prepare_game_features(
            game['home_team_id'],
            game['away_team_id'],
            game['game_time'],
            game['game_date']
        )

        if features is None:
            continue

        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]

        predicted_home_win = bool(prediction == 1)
        win_prob = float(max(probability))
        predicted_winner_id = game['home_team_id'] if predicted_home_win else game['away_team_id']
        predicted_winner = game['home_team'] if predicted_home_win else game['away_team']

        predictions.append({
            'scheduled_game_id': game['scheduled_game_id'],
            'game_date': game['game_date'],
            'home_team_id': game['home_team_id'],
            'away_team_id': game['away_team_id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'predicted_home_win': predicted_home_win,
            'predicted_winner_id': predicted_winner_id,
            'predicted_winner': predicted_winner,
            'win_probability': win_prob,
            'home_win_prob': float(probability[1])
        })

        print(
            f"  Predicted Winner: {predicted_winner} ({win_prob:.1%} confidence)")

    return pd.DataFrame(predictions)


def save_predictions_to_db(predictions_df, model_version='v1.0'):
    """Save predictions to database"""

    if predictions_df.empty:
        print("No predictions to save")
        return

    conn = get_db_connection()
    cur = conn.cursor()

    saved_count = 0

    for _, pred in predictions_df.iterrows():
        try:
            cur.execute("""
                INSERT INTO predictions (
                    scheduled_game_id,
                    home_team_id,
                    away_team_id,
                    game_date,
                    predicted_home_win,
                    predicted_winner_id,
                    win_probability,
                    model_version,
                    is_custom_matchup
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scheduled_game_id) DO UPDATE SET
                    predicted_home_win = EXCLUDED.predicted_home_win,
                    predicted_winner_id = EXCLUDED.predicted_winner_id,
                    win_probability = EXCLUDED.win_probability,
                    model_version = EXCLUDED.model_version
            """, (
                int(pred['scheduled_game_id']),
                int(pred['home_team_id']),
                int(pred['away_team_id']),
                pred['game_date'],
                pred['predicted_home_win'],
                int(pred['predicted_winner_id']),
                pred['win_probability'],
                model_version,
                False  # Not a custom matchup
            ))
            saved_count += 1
        except Exception as e:
            print(f"Error saving prediction: {e}")
            continue

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n✓ Saved {saved_count} predictions to database")


def display_predictions(predictions_df):
    """Display predictions in a nice format"""

    if predictions_df.empty:
        print("\nNo predictions to display")
        return

    print("\n" + "="*80)
    print("TODAY'S NBA GAME PREDICTIONS")
    print("="*80)

    for _, pred in predictions_df.iterrows():
        winner = pred['predicted_winner']
        loser = pred['away_team'] if pred['predicted_home_win'] else pred['home_team']
        prob = pred['win_probability']

        print(f"\n{pred['away_team']} @ {pred['home_team']}")
        print(f"  → Predicted Winner: {winner} ({prob:.1%})")
        print(f"  → Home Team Win Probability: {pred['home_win_prob']:.1%}")


def generate_predictions(date=None):
    """
    Main function to generate predictions
    Args:
        date: Optional date (YYYY-MM-DD). If None, uses today
    """
    print("="*80)
    print("NBA GAME PREDICTION GENERATOR")
    print("="*80)

    # 1. Load model
    print("\nStep 1: Loading model...")
    model = load_model()

    # 2. Get scheduled games
    print("\nStep 2: Loading scheduled games...")
    games_df = get_games_for_date(date)

    if games_df.empty:
        print("\n⚠ No scheduled games found for today")
        return

    # 3. Make predictions
    print("\nStep 3: Generating predictions...")
    predictions_df = make_predictions(model, games_df)

    # 4. Save to database
    print("\nStep 4: Saving predictions...")
    save_predictions_to_db(predictions_df)

    # 5. Display results
    display_predictions(predictions_df)

    print("\n" + "="*80)
    print("✓ Prediction generation complete!")
    print("="*80)

    return predictions_df


if __name__ == "__main__":
    # Generate predictions for today
    predictions = generate_predictions()

    # Or generate for specific date:
    # predictions = generate_predictions(date="2025-11-13")
