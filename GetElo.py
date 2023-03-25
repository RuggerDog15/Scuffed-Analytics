# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 06:35:18 2023

@author: mfloy
"""

from pathlib import Path
import warnings
from tqdm import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import time

datafolder = Path("./Data/")
rt = datafolder / "results_torn.csv"
results = pd.read_csv(rt)

results["date"] = pd.to_datetime(results["date"])
results = results.sort_values(by=["date"], ignore_index=True)
results["home_elo"] = 0
results["away_elo"] = 0
h_adv = 100
weight = 400

h = results["home_team"]
a = results["away_team"]
elo_teams = pd.concat([h, a], ignore_index=True).drop_duplicates().to_frame()
elo_teams = elo_teams.reset_index(drop=True)
elo_teams.rename(columns={elo_teams.columns[0]: "Team" }, inplace = True)
elo_teams = elo_teams.assign(ELOs = 1500)
elo_teams = elo_teams.sort_values(by="Team", ignore_index=True)

# ind = 0

for ind in tqdm(results.index):
    hometeam = results.loc[ind, "home_team"]
    awayteam = results.loc[ind, "away_team"]
    elo_home = elo_teams.loc[(elo_teams["Team"] == hometeam), "ELOs"].item()
    results.loc[ind, "home_elo"] = elo_home
    elo_away = elo_teams.loc[(elo_teams["Team"] == awayteam), "ELOs"].item()
    results.loc[ind, "away_elo"] = elo_away
    
    hs = results.loc[ind, "home_score"]
    aws = results.loc[ind, "away_score"]
    gd = hs - aws
    k = results.loc[ind, "Raw"]
    if abs(gd) < 2:
        k = k
    elif abs(gd) == 2:
        k = k * 1.5
    elif abs(gd) == 3:
        k = k * 1.75
    elif abs(gd) > 3:
        k = k * (abs(gd)+11)/8
    
    if gd == 0:
        wh = 0.5
        wa = 0.5
    elif gd > 0:
        wh = 1
        wa = 0
    elif gd < 0:
        wh = 0
        wa = 1
    
    if results.loc[ind, "neutral"] == True:
        h_adv = 0
        
    w = elo_home - elo_away + h_adv
    
    we = (1/(10 ** (-w/weight)+1))
    
    ht_prob = we
    at_prob = 1-ht_prob
    
    new_home = elo_home + (k * (wh-ht_prob))
    new_away = elo_away + (k * (wa-at_prob))
    elo_teams.loc[(elo_teams["Team"] == hometeam), "ELOs"] = new_home
    elo_teams.loc[(elo_teams["Team"] == awayteam), "ELOs"] = new_away
    

results.to_csv(datafolder / "results_elo.csv", index=False)
elo_teams.to_csv(datafolder / "team_elo.csv", index=False)
