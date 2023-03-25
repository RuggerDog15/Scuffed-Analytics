# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:28:28 2023

@author: mfloy
"""

import pandas as pd

from pathlib import Path
import warnings
from tqdm import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import time
from difflib import SequenceMatcher
import xgboost

datafolder = Path("./Data/")
file = datafolder / "conc.txt"
fixt = pd.read_csv(file, sep=",", header=None, names=["home_team", "away_team"])
fixt["neutral"] = False
fixt["Raw"] = 40

#load elo data
datafolder = Path("./Data/")
file = datafolder / "team_elo.csv"
elo_teams = pd.read_csv(file)

bad = []
predictions = pd.DataFrame(columns=["home_team","away_team"])
P_hat = pd.DataFrame()
P_all = pd.DataFrame()
#load the models
lst = ["home_score", "away_score"]
models = {}

for x in lst:
    model = xgboost.XGBClassifier()
    model.load_model(datafolder / f"model-{x}-multi.json")
    models[x] = model

#get elo scores
for ind in tqdm(fixt.index):
    hometeam = fixt.loc[ind, "home_team"].strip()
    awayteam = fixt.loc[ind, "away_team"].strip()
    
    if hometeam in elo_teams["Team"].values:
        elo_home = elo_teams.loc[(elo_teams["Team"] == hometeam), "ELOs"].item()
    else:
        elo_home = 0
        bad.append(hometeam)
    
    fixt.loc[ind, "home_elo"] = elo_home
    if awayteam in elo_teams["Team"].values:
        elo_away = elo_teams.loc[(elo_teams["Team"] == awayteam), "ELOs"].item()
    else:
        elo_away = 0
        bad.append(awayteam)
    fixt.loc[ind, "away_elo"] = elo_away
    neutral = fixt.loc[ind, "neutral"]
    raw = fixt.loc[ind, "Raw"]

    teams = {"home_team": [hometeam], "away_team": [awayteam],"home_elo": [elo_home], "away_elo": [elo_away], "neutral": [neutral], "Raw": [raw]}
    #teams = {"home_elo": [elo_home], "away_elo": [elo_away]}

    GameX = pd.DataFrame(teams)

    GameX["home_team"] = GameX["home_team"].astype("category")
    GameX["away_team"] = GameX["away_team"].astype("category")

    for x in lst:  
        P_hat[x] = models[x].predict(GameX)
    
    P_all = pd.concat([P_all, P_hat], ignore_index=True)


predictions = fixt.merge(P_all, left_index = True, right_index=True)


predictions.to_csv(datafolder / "cnl-score-predictions.csv", index=False)
