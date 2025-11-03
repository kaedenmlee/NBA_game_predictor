# NBA Game Predictor

**Author:** Kaeden Lee  
**Started:** October 24, 2025  
**Status:** In Development - Currently working on database and backend.

## Overview

An NBA game prediction system that analyzes historical data from multiple seasons (2020-21 through 2025-26) to predict game outcomes. The project combines web scraping, data processing, machine learning, and web deployment to create a comprehensive basketball analytics tool.

## Features

### Current Features
- **Multi-Season Data Collection**: Scrapes NBA game data from 2020-21 season to present
- **Comprehensive Statistics**: Collects 30+ statistics per game including:
  - Field goal percentages (FG%, 3P%, 2P%, FT%)
  - Rebounds (offensive, defensive, total)
  - Assists, steals, blocks, turnovers, and personal fouls
  - Team and opponent scores
  - Game metadata (date, time, attendance, game length)
- **Automated Data Cleaning**: Removes duplicate headers, filters incomplete games, and standardizes formats
- **Team-Based Analysis**: Tracks all 30 NBA teams with proper team names and abbreviations

### Planned Features
- Machine learning model for game outcome prediction
- Feature engineering (rolling averages, home/away advantage, rest days)
- Web interface for real-time predictions
- Database storage (PostgreSQL)
- REST API (Spring Boot)
- Historical accuracy tracking and model evaluation

## Technologies Used

### Programming Languages
- **Python 3.7+**: Data scraping and machine learning
- **Java**: Backend API development (planned)
- **SQL**: Database management (planned)

### Python Libraries
- **Web Scraping**
  - `beautifulsoup4`: HTML parsing and data extraction
  - `requests`: HTTP requests to Basketball Reference
  - `lxml`: HTML/XML parser for pandas
- **Data Processing**
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computing (planned)
- **Machine Learning** (planned)
  - `scikit-learn`: Model development
  - `tensorflow` or `pytorch`: Deep learning (optional)

### Frameworks & Tools (Planned)
- **Spring Boot**: Backend REST API
- **PostgreSQL**: Database for storing historical data and predictions
- **React** or **HTML/CSS/JavaScript**: Frontend interface

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### macOS Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd NBA_game_predictor
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install required libraries**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install beautifulsoup4 requests pandas lxml
```

### Requirements.txt
```
beautifulsoup4>=4.11.0
requests>=2.28.0
pandas>=1.5.0
lxml>=4.9.0
```

## Project Structure

```
NBA_game_predictor/
├── stat_scraping/
|   ├── initial_scraping_tests.py    # initial scraping tests
│   ├── final_scraping.py              # Main scraping script
│   ├── data_prediction.py       # Data preparation for ML
|   ├── preditction_results.csv   # prediction results
│   └── nba_scrape_data.csv  # Scraped data
├── README.md                    # Project documentation
└── venv/                        # Virtual environment (not in git)
```

## Usage

### Scraping Data

```bash
# Activate virtual environment
source venv/bin/activate

# Run the scraper
python stat_scraping/scraping.py
```

The scraper will:
1. Access Basketball Reference standings pages
2. Extract all team URLs
3. Scrape game logs and statistics for each team
4. Clean and format the data
5. Save to `nba_data_with_team_names.csv`

### Data Processing

```python
from fix_ordering import fix_game_ordering
import pandas as pd

# Load and process data
df = fix_game_ordering()

# Basic analysis
print(f"Total games: {len(df)}")
print(f"Teams: {df['Team_Name'].nunique()}")
print(f"Seasons: {df['Season'].unique()}")
```

## Data Source

All data is scraped from [Basketball Reference](https://www.basketball-reference.com/), a comprehensive basketball statistics website.

**Attribution:**
- Data provided by [Sports Reference LLC](https://www.sports-reference.com/)
- Website: https://www.basketball-reference.com/
- This project is for educational purposes only

**Scraping Ethics:**
- Respectful delays between requests (2-3 seconds)
- User-Agent headers included
- Follows robots.txt guidelines
- No excessive or abusive scraping

## Data Schema

### Cleaned Data Columns
_____________________________________________________________________________
| Column                   | Description                                    |
|--------------------------|------------------------------------------------|
| `Team_Name`              | Full team name (e.g., "Oklahoma City Thunder") |
| `Team_Abbr`              | Team abbreviation (e.g., "OKC")                | 
| `Season`                 | NBA season (e.g., "2024-25")                   |
| `Game_Number`            | Sequential game number in season               |
| `Date`                   | Game date                                      |
| `Start_Time`             | Game start time (ET)                           |
| `Opponent`               | Opposing team name                             |
| `Team_Score`             | Team's final score                             |
| `Opponent_Score`         | Opponent's final score                         |
| `Win`/`Loss`             | Win/Loss count                                 |
| `Field_Goals_Made`       | Field goals made                               |
| `Field_Goals_Attempted`  | Field goal attempts                            |
| `Three_Pointers_Made`    | Three-pointers made                            |
| `Free_Throws_Made`       | Free throws made                               |
| `Total_Rebounds`         | Total rebounds                                 |
| `Assists`                | Assists                                        |
| `Steals`                 | Steals                                         |
| `Blocks`                 | Blocks                                         |
| `Turnovers`              | Turnovers                                      |
_____________________________________________________________________________
*...and 15+ additional statistics*

## Challenges & Solutions

### Challenge 1: Hidden HTML Tables
**Problem:** Basketball Reference hides tables in HTML comments to prevent scraping  
**Solution:** Parse HTML comments and extract tables from commented code

### Challenge 2: Duplicate Header Rows
**Problem:** Long tables include repeated header rows for readability  
**Solution:** Filter rows where column values equal column names

### Challenge 3: Incomplete Current Season Data
**Problem:** Future games appear in tables but have no statistics  
**Solution:** Filter games where `LOG` (game length) is not null

### Challenge 4: Date Format Mismatches
**Problem:** Different date formats between game schedules and game logs  
**Solution:** Convert all dates to pandas datetime format before merging

## Development Roadmap

### Phase 1: Data Collection ✅
- [x] Scrape game schedules
- [x] Scrape game statistics
- [x] Clean and merge data
- [x] Handle multiple seasons

### Phase 2: Feature Engineering (In Progress)
- [ ] Calculate rolling averages (last 5, 10, 15 games)
- [ ] Home/away performance metrics
- [ ] Rest days between games
- [ ] Head-to-head records
- [ ] Strength of schedule

### Phase 3: Machine Learning
- [ ] Train/test split by season
- [ ] Feature selection
- [ ] Model training (Random Forest, XGBoost, Neural Networks)
- [ ] Cross-validation and hyperparameter tuning
- [ ] Model evaluation and accuracy metrics

### Phase 4: Backend Development
- [ ] PostgreSQL database setup
- [ ] Spring Boot REST API
- [ ] API endpoints for predictions
- [ ] Historical data storage

### Phase 5: Frontend & Deployment
- [ ] Web interface design
- [ ] Real-time prediction display
- [ ] Model accuracy dashboard
- [ ] Cloud deployment (AWS, Heroku, or similar)

## Contributing

This is a personal learning project, but suggestions and feedback are welcome! Feel free to open issues or reach out with ideas.

## License

This project is for educational purposes only. All NBA data belongs to the NBA and is provided by Basketball Reference.

## Acknowledgments

- **Basketball Reference** for providing comprehensive NBA statistics
- **Sports Reference LLC** for maintaining high-quality sports data
- Python community for excellent data science libraries

## Contact

**Kaeden Lee**  
GitHub: https://github.com/kaedenmlee
Email: kaedenmlee@gmail.com

---

*Last Updated: November 1, 2025*
