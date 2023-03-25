# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:33:32 2023

@author: mfloy
"""

import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm


readfolder = Path("./Data/")
bracket = pd.read_csv(readfolder / "WC22-Map.csv")
teams = pd.read_csv(readfolder / "WC22-Pretend-Group.csv")

for ind in tqdm(bracket.index):
    t1 = bracket.loc[ind, "Team1"]
    t2 = bracket.loc[ind, "Team2"]
    g = bracket.loc[ind, "Game"]
    team1 = teams.loc[(teams["Group"] == t1), "Team"].item()
    team2 = teams.loc[(teams["Group"] == t2), "Team"].item()
    #the random code below just randomly simulates the game results.  the scores or result test would come from the model
    score1 = random.randint(0, 100)
    score2 = random.randint(0, 100)
    if score1 >= score2:
        teams.loc[(teams["Group"] == t1), "Group"] = g+"W"
        teams.loc[(teams["Group"] == t2), "Group"] = g+"L"
    else:
        teams.loc[(teams["Group"] == t1), "Group"] = g+"L"
        teams.loc[(teams["Group"] == t2), "Group"] = g+"W"
    print(team1, team2, g)