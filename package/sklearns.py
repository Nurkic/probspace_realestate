# coding: utf-8

import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

train_path = "../input/train_data.csv"
test_path = "../input/test_data.csv"

""" load data"""
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

""" Preprocessing"""
import preprocess as pr

df = train["y"]

predata = pd.concat([train.drop("y", axis=1), test], ignore_index=True)
predata = pr.Preprocessor(predata).all("label")

prep_train = pd.concat([df, predata.iloc[:len(train), :]], axis=1)
prep_test = predata.iloc[len(train):, :]

# prep_train = pr.Preprocessor(prep_train).rename_t()
# prep_test = pr.Preprocessor(prep_test).rename_t()

train_X = prep_train.drop(["y", "id"], axis=1)
train_y = prep_train["y"]



""" model train & predict"""
reg = OGBMRegressor(random_state=71)
reg.fit(train_X, train_y)

res = reg.predict(prep_test.drop(["id"], axis=1))

""" export submit file"""
result = pd.DataFrame(test.index, columns=["id"])
result["y"] = res
result.to_csv("../output/result_realestate.csv", index=False)