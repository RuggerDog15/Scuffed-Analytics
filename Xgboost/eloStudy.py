# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:52:08 2023

@author: mfloy
"""

from pathlib import Path
import warnings
from tqdm import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import time

readfolder = Path("./Data/")
results = pd.read_csv(readfolder / "results_elo.csv")
shoot = pd.read_csv(readfolder / "shootouts.csv")

results['date']= pd.to_datetime(results['date'])
shoot['date']= pd.to_datetime(shoot['date'])

for ind in tqdm(shoot.index):
    dt = shoot.loc[ind, "date"]
    hm = shoot.loc[ind, "home_team"]
    test = results.loc[(results["date"] == dt) & (results["home_team"] == hm)]
    if test.empty == False:
        elo_home = results.loc[(results["date"] == dt) & (results["home_team"] == hm), "home_elo"].item()
        shoot.loc[ind, "home_elo"] = elo_home
        elo_away = results.loc[(results["date"] == dt) & (results["home_team"] == hm), "away_elo"].item()
        shoot.loc[ind, "away_elo"] = elo_away
        shoot.loc[ind, "elo_diff"] = elo_home - elo_away
        if shoot.loc[ind, "winner"] == hm:
            shoot.loc[ind, "result"] = 1
        else:
            shoot.loc[ind, "result"] = -1
    else:
        shoot.loc[ind, "home_elo"] = "Unknown"
        shoot.loc[ind, "away_elo"] = "Unknown"
        shoot.loc[ind, "elo_diff"] = "Unknown"
        
shoot = shoot.loc[(shoot["home_elo"] != "Unknown")]
shoot["result_plus"] = shoot["elo_diff"] * shoot["result"]
high = shoot[shoot.result_plus > 0]
low = shoot[shoot.result_plus < 0]
highwins = len(high)
lowwins = len(low)
highavg = abs(high["elo_diff"]).mean()
lowavg = abs(low["elo_diff"]).mean()
highwinper = highwins / (highwins + lowwins)
lowwinper = lowwins / (highwins + lowwins)

elo_shoot = pd.DataFrame()
elo_shoot.loc[0, "SO High Wins"] = highwins
elo_shoot.loc[0,"SO High Win Percentage"] = highwinper
elo_shoot.loc[0,"SO Average Elo Diff High Win"] = highavg
elo_shoot.loc[0,"SO Low Wins"] = lowwins
elo_shoot.loc[0,"SO Low Win Percentage"] = lowwinper
elo_shoot.loc[0,"SO Average Elo Diff Low Win"] = lowavg


for ind in tqdm(results.index):
    elo_diff = results.loc[ind, "home_elo"] - results.loc[ind, "away_elo"]
    results.loc[ind, "elo_diff"] = elo_diff
    gd = results.loc[ind, "home_score"] - results.loc[ind, "away_score"]
    results.loc[ind, "goal_diff"] = gd

results["result_plus"] = results["elo_diff"] * results["goal_diff"]
high = results[results.result_plus > 0]
low = results[results.result_plus < 0]
draw = results[results.result_plus == 0]
highwins = len(high)
lowwins = len(low)
drawnum = len(draw)
highavg = abs(high["elo_diff"]).mean()
lowavg = abs(low["elo_diff"]).mean()
drawavg = abs(draw["elo_diff"]).mean()
highwinper = highwins / (highwins + lowwins + drawnum)
lowwinper = lowwins / (highwins + lowwins + drawnum)
drawper = drawnum / (highwins + lowwins + drawnum)

elo_shoot.loc[0,"GM High Wins"] = highwins
elo_shoot.loc[0,"GM High Win Percentage"] = highwinper
elo_shoot.loc[0,"GM Average Elo Diff High Win"] = highavg
elo_shoot.loc[0,"GM Low Wins"] = lowwins
elo_shoot.loc[0,"GM Low Win Percentage"] = lowwinper
elo_shoot.loc[0,"GM Average Elo Diff Low Win"] = lowavg
elo_shoot.loc[0,"GM Draws"] = drawnum
elo_shoot.loc[0,"GM Draw Percentage"] = drawper
elo_shoot.loc[0,"GM Average Elo Diff Draw"] = drawavg

tot = highwins + lowwins + drawnum

elo_shoot.to_csv( readfolder / "Elo-Study.csv", index=False)
