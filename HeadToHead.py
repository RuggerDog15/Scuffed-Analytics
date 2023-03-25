# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:47:47 2023

@author: mfloy
"""

from pathlib import Path
import warnings
from tqdm import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

datafolder = Path("./Data/")
rt = datafolder / "results_elo.csv"
results = pd.read_csv(rt)

team1 = "Grenada"
team2 = "United States"
#raw = 30

home = results.loc[(results["home_team"] == team1) & (results["away_team"] == team2)]
away = results.loc[(results["home_team"] == team2) & (results["away_team"] == team1)]

games = pd.concat([home, away], ignore_index=True)

gms1 = results.loc[(results["home_team"] == team1)]
gms = results.loc[(results["away_team"] == team1)]
gms = pd.concat([gms, gms1], ignore_index=True)