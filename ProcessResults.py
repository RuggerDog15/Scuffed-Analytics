# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:22:29 2023

@author: mfloy
"""

import os
from pathlib import Path
import warnings
from tqdm import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import time
import datetime

w_factor = .05
dataIn = Path("D:\Python\SoccerGitData\international_results")
rt = dataIn / "results.csv"
results = pd.read_csv(rt)

home_score =  (results["home_score"] - results["away_score"]).tolist()

home_loss = []
home_draw = []
home_win = []

for X in home_score:
    if X > 0:
        home_win.append(1)
    else:
        home_win.append(0)
        
for X in home_score:
    if X == 0:
        home_draw.append(1)
    else:
        home_draw.append(0)
        
for X in home_score:
    if X < 0:
        home_loss.append(1)
    else:
        home_loss.append(0)

results["home_win"] = home_win
results["home_loss"] = home_loss
results["home_draw"] = home_draw

results["year"] = pd.DatetimeIndex(results["date"]).year
results["weight_date"] = (1 + ((results["year"]-1872) * w_factor))

dataOut = Path("./Data")

rout = os.path.join(dataOut, "results-out.csv")
results.to_csv(rout, index=False)

#torns = results["tournament"].drop_duplicates()

torns = pd.read_csv(dataOut / "torns-out.csv")

results_torn = torns.merge(results, on="tournament")

missingRaw = results.loc[results_torn["Raw"] == ""]

results_torn["weight"] =  (results_torn["Modified"] * results_torn["weight_date"])

results_torn.to_csv(dataOut / "results_torn.csv", index=False)

#create initial elo table
