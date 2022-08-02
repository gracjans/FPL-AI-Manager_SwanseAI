import datetime

import pandas as pd


def idx_to_team_name(master_team_list: pd.DataFrame, team: int, season: str):
    # get team name from master_team_list with appropriate season and team
    return master_team_list[(master_team_list['season'] == season) & (master_team_list['team'] == team)]['team_name'].values[0]


def str_date_months_back(date: str, months_back: int):
    # get date months back from date
    return (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=months_back * 30)).strftime('%Y-%m-%d')


def str_date_days_forward(date: str, days_forward: int):
    # get date months back from date
    return (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=days_forward)).strftime('%Y-%m-%d')


def rename_teams(teams_column: pd.Series):
    # Rename the teams name, that they are compatible with the teams names from Understat data.
    teams_column.replace('Man City', 'Manchester City', inplace=True)
    teams_column.replace('Man Utd', 'Manchester United', inplace=True)
    teams_column.replace('Spurs', 'Tottenham', inplace=True)
    teams_column.replace('West Brom', 'West Bromwich Albion', inplace=True)
    teams_column.replace('Newcastle', 'Newcastle United', inplace=True)
    teams_column.replace('Wolves', 'Wolverhampton Wanderers', inplace=True)
    teams_column.replace('Sheffield Utd', 'Sheffield United', inplace=True)

    return teams_column
