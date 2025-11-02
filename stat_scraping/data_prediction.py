import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


def prepare_data(scrape_file):
    df_sorted = pd.read_csv(scrape_file)
    df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
    df_sorted["Home"] = df_sorted["Home"].astype("category").cat.codes
    df_sorted["opp_code"] = df_sorted["Opponent"].astype("category").cat.codes
    df_sorted["team_code"] = df_sorted["Team_Name"].astype("category").cat.codes
    df_sorted["hour"] = df_sorted["Start (ET)"].str.extract(r'(\d+)')[0].astype(int)
    df_sorted["day_code"] = df_sorted["Date"].dt.dayofweek
    df_sorted["target"] = (df_sorted["Tm"] > df_sorted["Opp"]).astype(int)
    df = df_sorted.sort_values(["Team_Name", "Date"]).reset_index(drop=True)
    return df

def add_rolling_averages(df, window = 10):
    cols = ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF"]
    new_cols = [f"{c}_rolling" for c in cols]
    
    def caclulate_rolling(group):
        group = group.sort_values("Date")
        rolling_stats = group[cols].rolling(window, closed = "left").mean()
        group[new_cols] = rolling_stats
        return group
    
    df = df.groupby(["Team_Name", "Season"], group_keys = False).apply(caclulate_rolling)
    
    df = df.dropna(subset = new_cols)
    
    return df, new_cols
    
def train_and_evaluate(df, predictors, test_date = "2025-04-13"):
    train = df[df["Date"] < test_date]
    test = df[df["Date"] > test_date]
    
    print(f"\nTraining set: {len(train)} games")
    print(f"Test set: {len(test)} games")
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        min_samples_split=10, 
        random_state=1,
        n_jobs= -1
    )
    
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    
    accuracy = accuracy_score(test["target"], preds)
    precision = precision_score(test["target"], preds)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    
    results = pd.DataFrame({
        "Date": test["Date"].values,
        "Team": test["Team_Name"].values,
        "Opponent": test["Opponent"].values,
        "Home": test["Home"].values,
        "Actual_Won": test["target"].values,
        "Predicted_Won": preds,
        "Team_Score": test["Tm"].values,
        "Opponent_Score": test["Opp"].values
    })
    
    results["Correct"] = (results["Actual_Won"] == results["Predicted_Won"]).astype(int)
    
    return results, accuracy, precision, rf

def analyze_feature_importance(rf, predictors):
    importances = pd.DataFrame({
        "feature": predictors,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending = False)
    
    print("\n=== Feature Importance ===")
    importances = importances.sort_values(by = "importance", ascending = False)
    print(importances.head(23))
    return importances

def main():
    print("=== NBA Game Prediction Pipeline ===\n")
    
    df = prepare_data("nba_scrape_data.csv")
    
    df, rolling_cols = add_rolling_averages(df, window = 10)
    
    base_predictors = ["Home", "opp_code", "team_code", "hour", "day_code"]
    all_predictors = base_predictors + rolling_cols
    
    results, accuracy, precision, rf = train_and_evaluate(df, all_predictors, test_date="2025-04-13")
    
    analyze_feature_importance(rf, all_predictors)
    
    print(results.head(10))
    
    results.to_csv("prediction_results.csv", index = False)
    
    return df, results, rf
    

if __name__ == "__main__":
    df, results, rf = main()
    
