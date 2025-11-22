import pandas as pd
import psycopg2
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        dbname="nba_stats",
        user="kaedenlee",
        host="localhost"
    )


def load_data_from_database():
    """Load game history with team IDs from PostgreSQL"""
    conn = get_db_connection()

    query = """
        SELECT 
            gh.*,
            t.team_id,
            opp.team_id as opponent_team_id
        FROM game_history gh
        JOIN teams t ON gh.team_name = t.team_name
        JOIN teams opp ON gh.opponent = opp.team_name
        WHERE gh.season IN ('2021-22', '2022-23', '2023-24', '2024-25', '2025-26')
        ORDER BY gh.game_date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} games from database")
    return df


def prepare_data(df):
    """Prepare features for model training"""

    # Convert date
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Create categorical codes
    df['opp_code'] = df['opponent'].astype('category').cat.codes
    df['team_code'] = df['team_name'].astype('category').cat.codes

    # Extract hour from start time
    df['hour'] = df['start_time'].str.extract(r'(\d+)')[0].astype(float)
    df['hour'] = df['hour'].fillna(19)  # Default to 7pm if missing

    # Day of week
    df['day_code'] = df['game_date'].dt.dayofweek

    # Target: 1 if team won, 0 if lost
    df['target'] = (df['team_score'] > df['opp_score']).astype(int)

    # Sort by team and date
    df = df.sort_values(['team_name', 'game_date']).reset_index(drop=True)

    print(f"Prepared {len(df)} games for training")
    return df


def add_rolling_stats_from_db(df):
    """
    Use end-of-season rolling stats for each team per season
    Much simpler - just use the final stats from each season
    """
    conn = get_db_connection()

    # Get end-of-season rolling stats for each team/season
    rolling_query = """
        WITH ranked_stats AS (
            SELECT 
                team_id,
                season,
                as_of_date,
                avg_points,
                avg_fg_pct,
                avg_3p_pct,
                avg_ft_pct,
                avg_rebounds,
                avg_assists,
                avg_steals,
                avg_blocks,
                avg_turnovers,
                home_win_pct,
                away_win_pct,
                last_5_wins,
                ROW_NUMBER() OVER (PARTITION BY team_id, season ORDER BY as_of_date DESC) as rn
            FROM team_rolling_stats
        )
        SELECT * FROM ranked_stats WHERE rn = 1
    """

    rolling_df = pd.read_sql(rolling_query, conn)
    conn.close()

    print(f"Loaded rolling stats for {len(rolling_df)} team-seasons")

    # Merge rolling stats with games
    df_merged = df.merge(
        rolling_df[['team_id', 'season', 'avg_points', 'avg_fg_pct', 'avg_3p_pct',
                   'avg_ft_pct', 'avg_rebounds', 'avg_assists', 'avg_steals',
                    'avg_blocks', 'avg_turnovers', 'home_win_pct', 'away_win_pct', 'last_5_wins']],
        on=['team_id', 'season'],
        how='left'
    )

    # Rename to match expected column names
    df_merged = df_merged.rename(columns={
        'avg_points': 'rolling_avg_points',
        'avg_fg_pct': 'rolling_avg_fg_pct',
        'avg_3p_pct': 'rolling_avg_3p_pct',
        'avg_ft_pct': 'rolling_avg_ft_pct',
        'avg_rebounds': 'rolling_avg_rebounds',
        'avg_assists': 'rolling_avg_assists',
        'avg_steals': 'rolling_avg_steals',
        'avg_blocks': 'rolling_avg_blocks',
        'avg_turnovers': 'rolling_avg_turnovers'
    })

    # Drop rows with missing rolling stats
    rolling_cols = [
        'rolling_avg_points', 'rolling_avg_fg_pct', 'rolling_avg_3p_pct',
        'rolling_avg_ft_pct', 'rolling_avg_rebounds', 'rolling_avg_assists',
        'rolling_avg_steals', 'rolling_avg_blocks', 'rolling_avg_turnovers',
        'home_win_pct', 'away_win_pct', 'last_5_wins'
    ]

    df_final = df_merged.dropna(subset=rolling_cols).reset_index(drop=True)

    print(f"Added rolling stats: {len(df_final)} games remaining")
    return df_final, rolling_cols


def train_and_evaluate(df, predictors, test_date="2025-01-01"):
    """Train model and evaluate performance"""

    # Split by date
    train = df[df['game_date'] < test_date]
    test = df[df['game_date'] >= test_date]

    print(f"\n{'='*60}")
    print(f"Training set: {len(train)} games (before {test_date})")
    print(f"Test set: {len(test)} games (from {test_date} onward)")
    print(f"{'='*60}")

    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=10,
        max_depth=15,
        random_state=1,
        n_jobs=-1
    )

    rf.fit(train[predictors], train['target'])

    # Make predictions
    preds = rf.predict(test[predictors])

    # Calculate metrics
    accuracy = accuracy_score(test['target'], preds)
    precision = precision_score(test['target'], preds)

    print(f"\n{'='*60}")
    print(f"Model Performance:")
    print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"{'='*60}")

    # Create results dataframe
    results = pd.DataFrame({
        'Date': test['game_date'].values,
        'Team': test['team_name'].values,
        'Opponent': test['opponent'].values,
        'Home': test['home_status'].values,
        'Actual_Won': test['target'].values,
        'Predicted_Won': preds,
        'Team_Score': test['team_score'].values,
        'Opponent_Score': test['opp_score'].values
    })

    results['Correct'] = (results['Actual_Won'] ==
                          results['Predicted_Won']).astype(int)

    return results, accuracy, precision, rf


def analyze_feature_importance(rf, predictors):
    """Analyze which features are most important"""

    importances = pd.DataFrame({
        'feature': predictors,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*60}")
    print("Feature Importance (Top 15):")
    print(f"{'='*60}")
    for idx, row in importances.head(15).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    return importances


def save_model(model, filename='models/nba_model.pkl'):
    """Save trained model to disk"""
    import os

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✓ Model saved to {filename}")


def predict_game(model, team_stats, opponent_stats, is_home, predictors):
    """
    Predict outcome of a single game

    Args:
        model: Trained RandomForestClassifier
        team_stats: dict with team's rolling stats
        opponent_stats: dict with opponent's rolling stats
        is_home: 1 if team is home, 0 if away
        predictors: list of feature names

    Returns:
        dict with prediction and probability
    """

    # Create feature vector
    features = pd.DataFrame([{
        'home_status': is_home,
        'opp_code': team_stats.get('opp_code', 0),
        'team_code': team_stats.get('team_code', 0),
        'hour': team_stats.get('hour', 19),
        'day_code': team_stats.get('day_code', 0),
        'rolling_avg_points': team_stats.get('rolling_avg_points', 110),
        'rolling_avg_fg_pct': team_stats.get('rolling_avg_fg_pct', 0.45),
        'rolling_avg_3p_pct': team_stats.get('rolling_avg_3p_pct', 0.35),
        'rolling_avg_ft_pct': team_stats.get('rolling_avg_ft_pct', 0.75),
        'rolling_avg_rebounds': team_stats.get('rolling_avg_rebounds', 45),
        'rolling_avg_assists': team_stats.get('rolling_avg_assists', 25),
        'rolling_avg_steals': team_stats.get('rolling_avg_steals', 7),
        'rolling_avg_blocks': team_stats.get('rolling_avg_blocks', 5),
        'rolling_avg_turnovers': team_stats.get('rolling_avg_turnovers', 14),
        'home_win_pct': team_stats.get('home_win_pct', 0.5) if is_home else team_stats.get('away_win_pct', 0.5),
        'away_win_pct': opponent_stats.get('away_win_pct', 0.5) if is_home else opponent_stats.get('home_win_pct', 0.5),
        'last_5_wins': team_stats.get('last_5_wins', 2),
    }])

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        'predicted_winner': int(prediction),
        'win_probability': float(max(probability)),
        'home_team_win_prob': float(probability[1]) if is_home else float(probability[0])
    }


def main():
    """Main training pipeline"""

    print("="*60)
    print("NBA Game Prediction Model - PostgreSQL Version")
    print("="*60)

    # 1. Load data from database
    print("\nStep 1: Loading data from PostgreSQL...")
    df = load_data_from_database()

    # 2. Prepare features
    print("\nStep 2: Preparing features...")
    df = prepare_data(df)

    # 3. Add rolling stats from database
    print("\nStep 3: Adding rolling statistics...")
    df, rolling_cols = add_rolling_stats_from_db(df)

    # 4. Define predictors
    base_predictors = ['home_status', 'opp_code',
                       'team_code', 'hour', 'day_code']
    all_predictors = base_predictors + rolling_cols

    print(f"\nUsing {len(all_predictors)} features for prediction")

    # 5. Train and evaluate
    print("\nStep 4: Training model...")
    results, accuracy, precision, rf = train_and_evaluate(
        df,
        all_predictors,
        test_date="2024-10-01"  # Train on 2021-2024, test on 2024-25 and 2025-26
    )

    # 6. Analyze feature importance
    analyze_feature_importance(rf, all_predictors)

    # 7. Save model
    save_model(rf)

    # 8. Show sample predictions
    print(f"\n{'='*60}")
    print("Sample Predictions:")
    print(f"{'='*60}")
    print(results.head(10).to_string(index=False))

    # 9. Save results
    results.to_csv('prediction_results.csv', index=False)
    print(f"\n✓ Results saved to prediction_results.csv")

    return df, results, rf


if __name__ == "__main__":
    df, results, model = main()
