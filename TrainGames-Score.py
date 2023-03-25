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

import xgboost

fx = ["home_team", "away_team", "home_elo", "away_elo", "neutral", "Raw"]
#fx = ["home_elo", "away_elo"]
#fy = ["home_win", "home_loss", "home_draw"]
#fy = ["home_win", "home_loss"]
fy = ["home_score", "away_score"]
snum = 0
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

from sklearn.preprocessing import LabelEncoder

Y_hat = pd.DataFrame()
models = {}
for col in tqdm(list(Y.columns)):
    le = LabelEncoder()
    Yl = le.fit_transform(Y[col])
    model = xgboost.XGBClassifier(tree_method="gpu_hist", n_estimators=32, max_depth=16, objective="multi:softprob", eval_metric='logloss', enable_categorical=True, max_cat_to_onehot=1)
    model.fit(X, Yl, sample_weight=weight)
    #model.fit(X, Y[col])
    models[col] = model
    model.save_model(datafolder / f"model-{col}-multi.json")
    
# models.save_model(datafolder / "model-cat.json")
    
stop_proc, stop_time = time.process_time(), time.perf_counter()

print("Process time ", stop_proc-start_proc, "  Elapsed Time ", stop_time-start_time)    
    
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split

testX, testY = X, Y

xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.15)

"""def evaluate(y, y_hat):
    p = sum(y) / len(y)
    base = [p] * len(y)
    brier = brier_score_loss(y, y_hat)
    print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
    ll = log_loss(y, y_hat)
    print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))"""

start_proc, start_time = time.process_time(), time.perf_counter()

for col in testY.columns:
    Y_hat[col] = models[col].predict(testX)
    print(f"### Y: {col} ###")
    tY = testY[col].to_numpy()
    Yh = Y_hat[col].to_numpy()
    a = accuracy_score(tY, Yh)
    print(f"Accuracy Scores {a}")


    
stop_proc, stop_time = time.process_time(), time.perf_counter()
print("Process time ", stop_proc-start_proc, "  Elapsed Time ", stop_time-start_time)   

hometeam = "Grenada"
awayteam = "United States"
elo_home = elo_teams.loc[(elo_teams["Team"] == hometeam), "ELOs"].item()
elo_away = elo_teams.loc[(elo_teams["Team"] == awayteam), "ELOs"].item()

teams = {"home_team": [hometeam], "away_team": [awayteam],"home_elo": [elo_home], "away_elo": [elo_away], "neutral": False, "Raw": 40}
#teams = {"home_elo": [elo_home], "away_elo": [elo_away]}

GameX = pd.DataFrame(teams)

GameX["home_team"] = GameX["home_team"].astype("category")
GameX["away_team"] = GameX["away_team"].astype("category")
P_hat = pd.DataFrame()

for col in testY.columns:  
    P_hat[col] = models[col].predict(GameX) 
    