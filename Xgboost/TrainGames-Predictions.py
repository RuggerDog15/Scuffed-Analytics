# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:22:29 2023

@author: mfloy
"""

from pathlib import Path
import warnings
from tqdm import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import time

datafolder = Path("./Data/")
rt = datafolder / "results_elo.csv"
results = pd.read_csv(rt)
re = datafolder / "team_elo.csv"
elo_teams = pd.read_csv(re)
file = datafolder / "conc.txt"
fixt = pd.read_csv(file, sep=",", header=None, names=["home_team", "away_team"])

import xgboost

fx = ["home_team", "away_team", "home_elo", "away_elo"]
#fx = ["home_elo", "away_elo"]
fy = ["home_win", "home_loss", "home_draw"]
#fy = ["home_win", "home_loss"]
#fy = ["home_score", "away_score"]
snum = 1000
weight = results["weight"]
weight = weight.loc[snum:]
weight = weight.to_list()


X = results[(fx)]
Y = results[(fy)]
X = X.loc[snum:]
Y = Y.loc[snum:]

X["home_team"] = X["home_team"].astype("category")
X["away_team"] = X["away_team"].astype("category")

start_proc, start_time = time.process_time(), time.perf_counter()

Y_hat = pd.DataFrame()
models = {}
for col in tqdm(list(Y.columns)):
    model = xgboost.XGBClassifier(tree_method="gpu_hist", n_estimators=32, max_depth=32, eval_metric='logloss', enable_categorical=True, max_cat_to_onehot=1)
    model.fit(X, Y[col], sample_weight=weight)
    models[col] = model
    model.save_model(datafolder / f"model-{col}.json")
    
# models.save_model(datafolder / "model-cat.json")
    
stop_proc, stop_time = time.process_time(), time.perf_counter()

print("Process time ", stop_proc-start_proc, "  Elapsed Time ", stop_time-start_time)    
    
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split

testX, testY = X, Y

xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.15)

def evaluate(y, y_hat):
    p = sum(y) / len(y)
    base = [p] * len(y)
    brier = brier_score_loss(y, y_hat)
    print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
    ll = log_loss(y, y_hat)
    print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

start_proc, start_time = time.process_time(), time.perf_counter()

for col in testY.columns:
    Y_hat[col] = [p[1] for p in models[col].predict_proba(testX)]
    print(f"### Y: {col} ###")
    evaluate(testY[col], Y_hat[col])
    cv_score = cross_val_score(model, xtrain, ytrain, cv=5)
    print("  Mean cross-validation score: %.2f" % cv_score.mean())
    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
    print("  K-fold CV average score: %.2f" % kf_cv_scores.mean())
    
stop_proc, stop_time = time.process_time(), time.perf_counter()
print("Process time ", stop_proc-start_proc, "  Elapsed Time ", stop_time-start_time)   

hometeam = "United States"
awayteam = "El Salvador"
elo_home = elo_teams.loc[(elo_teams["Team"] == hometeam), "ELOs"].item()
elo_away = elo_teams.loc[(elo_teams["Team"] == awayteam), "ELOs"].item()

teams = {"home_team": [hometeam], "away_team": [awayteam],"home_elo": [elo_home], "away_elo": [elo_away]}
#teams = {"home_elo": [elo_home], "away_elo": [elo_away]}

GameX = pd.DataFrame(teams)

GameX["home_team"] = GameX["home_team"].astype("category")
GameX["away_team"] = GameX["away_team"].astype("category")
P_hat =pd.DataFrame()

for col in testY.columns:  
    P_hat[col] = [p[1] for p in models[col].predict_proba(GameX)]
    
bad = []
predictions = pd.DataFrame(columns=["home_team","away_team"])
P_all = pd.DataFrame()

#get elo scores
for ind in tqdm(fixt.index):
    hometeam = fixt.loc[ind, "home_team"].strip()
    awayteam = fixt.loc[ind, "away_team"].strip()
    #hometeam = "Grenada"
    #awayteam = "United States"
    
    """if hometeam in elo_teams["Team"].values:
        elo_home = elo_teams.loc[(elo_teams["Team"] == hometeam), "ELOs"].item()
    else:
        elo_home = 0
        bad.append(hometeam)
    
    fixt.loc[ind, "home_elo"] = elo_home
    if awayteam in elo_teams["Team"].values:
        elo_away = elo_teams.loc[(elo_teams["Team"] == awayteam), "ELOs"].item()
    else:
        elo_away = 0
        bad.append(awayteam)"""   

    elo_home = elo_teams.loc[(elo_teams["Team"] == hometeam), "ELOs"].item()
    elo_away = elo_teams.loc[(elo_teams["Team"] == awayteam), "ELOs"].item()
    #fixt.loc[ind, "away_elo"] = elo_away
   
    teams = {"home_team": [hometeam], "away_team": [awayteam],"home_elo": [elo_home], "away_elo": [elo_away]}
    #teams = {"home_elo": [elo_home], "away_elo": [elo_away]}
   
    GameX = pd.DataFrame(teams)
   
    GameX["home_team"] = GameX["home_team"].astype("category")
    GameX["away_team"] = GameX["away_team"].astype("category")
    P_hat =pd.DataFrame()
   
    for col in testY.columns:  
        P_hat[col] = [p[1] for p in models[col].predict_proba(GameX)]
    
    P_all = pd.concat([P_all, P_hat], ignore_index=True)


predictions = fixt.merge(P_all, left_index = True, right_index=True)
    