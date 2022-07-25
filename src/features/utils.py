import datetime


def idx_to_team_name(master_team_list, team, season):
    # get team name from master_team_list with appropriate season and team
    return master_team_list[(master_team_list['season'] == season) & (master_team_list['team'] == team)]['team_name'].values[0]


def str_date_months_back(date, months_back):
    # get date months back from date
    return (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
