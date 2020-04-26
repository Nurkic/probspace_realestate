# coding: utf-8

import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
from datetime import datetime
import tensorflow as tf

train_path = "../input/train_data.csv"
test_path = "../input/test_data.csv"

""" load raw data"""
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

""" Preprocessing"""
import preprocess as pr
import impute as im

import copy

df = train["y"]

predata = pd.concat([train.drop("y", axis=1), test], ignore_index=True)
predata_copy = copy.deepcopy(predata)
predata_onehot = pr.Preprocessor(predata).all("onehot")
predata_label = pr.Preprocessor(predata_copy).all("label")

""" missing values imputation"""
num_list = [
    "TimeToNearestStation", "TotalFloorArea", "Area", "Frontage", "BuildingYear", "BuildingAge", 
    "Breadth", "CoverageRatio", "FloorAreaRatio", "Period"
    ]
predata_onehot = im.Imputer(predata_onehot).num_imputer(num_list)
predata_onehot = predata_onehot.drop(['Type','Region','MunicipalityCode','Prefecture','Municipality','DistrictName','NearestStation',
            'FloorPlan','LandShape','Structure','Use','Purpose','Classification','CityPlanning', 'Direction',
            'Renovation','Remarks','era_name'], axis=1)

prep_train_onehot = pd.concat([df, predata_onehot.iloc[:len(train), :]], axis=1)
prep_test_onehot = predata_onehot.iloc[len(train):, :]

prep_train_label = pd.concat([df, predata_label.iloc[:len(train), :]], axis=1)
prep_test_label = predata_label.iloc[len(train):, :]


""" define data"""
train_label = prep_train_label.drop(["y", "id", "Prefecture", "Municipality"], axis=1)
train_y_label = prep_train_label["y"]
test_label = prep_test_label.drop(["id", "Prefecture", "Municipality"], axis=1)

train_onehot = prep_train_onehot.drop(["y", "id"], axis=1).values
train_y_onehot = df.values
test_onehot = prep_test_onehot.drop(["id"], axis=1).values


""" target encoding"""
"""from feature_selection import FeatureSelector as FS, cross_validator
train_X_te, test_X_te = FS(train_X, train_y).target_encoder(test_X)"""

""" feature selection"""
"""selected = FS(train_X, train_y).greedy_forward_selection()"""
"""selected_te = FS(train_X_te, train_y).greedy_forward_selection()"""
"""print("selected features:"+ str(selected))"""
"""print("selected target encoding features:"+ str(selected_te))"""

""" check cross validation score"""
"""cv1 = cross_validator(train_X, train_y)
cv2 = cross_validator(train_X[selected], train_y)
cv3 = cross_validator(train_X_te, train_y)
cv4 = cross_validator(train_X_te[selected_te], train_y)

print("base rmse:"+ str(cv1))
print("feature_selected rmse:"+ str(cv2))
print("target encoding rmse:"+ str(cv3))
print("target encoding and feature selection rmse:"+ str(cv4))"""


""" define NN Class"""
class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dense3 = tf.keras.layers.Dense(1028, activation="relu")
        self.dense4 = tf.keras.layers.Dense(1, activation="relu")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout1(x)
        x = self.dense3(x)
        return self.dense4(x)


""" cross validation"""
def cross_validator_lgbm(
    train_X: pd.DataFrame, 
    train_y: pd.DataFrame
    ):
    rmses = []
     
    oof = np.zeros((len(train_y), ))
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    for tr_idx, va_idx in kf.split(train_X):
        tr_X, va_X = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        reg = OGBMRegressor(random_state=71)
        reg.fit(tr_X, tr_y)
        va_pred = reg.predict(va_X)
            
        rmse = np.sqrt(mean_squared_error(va_y, va_pred))
            
        rmse.append(rmse)
            
        oof[va_idx] = va_pred
    return np.mean(rmses), oof

def cross_validator_nn(
    train_X: pd.DataFrame, 
    train_y: pd.DataFrame
    ):
    rmses = []
        
    oof = np.zeros((len(train_y), ))
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    for tr_idx, va_idx in kf.split(train_X):
        tr_X, va_X = train_X[tr_idx], train_X[va_idx]
        tr_y, va_y = train_y[tr_idx], train_y[va_idx]
            
        
    model = MLP()
    tensorboard = TensorBoard(log_dir="logs")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "./checkpoint/MLP-{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=True,
        period=200
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-1),
        loss='mean_squared_error',
        metrics=[
            "mean_squared_error", 
        ]
    )
    model.fit(
        tr_X, 
        tr_y, 
        epochs=100,
        batch_size=200,
        validation_data=(va_X, va_y),
        callbacks=[tensorboard, checkpoint]
    )
            
    """val_mse, val_acc = model.evaluate(x_valid,  y_valid)"""
    """rmse = np.sqrt(val_rmse)"""
    va_pred = model.predict_on_batch(va_X)
    rmse = np.sqrt(mean_squared_error(va_y, va_pred))
            
    rmse.append(rmse)
            
    oof[va_idx] = va_pred

    return np.mean(rmses), oof



rmse_lgbm, oof_lgbm = cross_validator_lgbm(train_label, train_y_label)
print("lgbm root mean squared error : " + str(rmse_lgbm))

rmse_nn, oof_nn = cross_validator_nn(train_onehot, train_y_onehot)
print("nn root mean squared error : " + str(rmse_nn))


""" train lightGBM"""
model_lgbm = OGBMRegressor(random_state=71)
model_lgbm.fit(train_label, train_y_label)

""" check feature importances"""
"""importances = pd.DataFrame(
    reg.feature_importances_, index=train_X.columns, 
    columns=["importance"]
    )
importances = importances.sort_values("importance",
    ascending=False
    )

print(importances)"""


""" train NN"""
model_nn = MLP()
tensorboard = TensorBoard(log_dir="logs")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "./checkpoint/MLP-{epoch:04d}.ckpt",
    verbose=1,
    save_weights_only=True,
    period=300
)
model_nn.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-1),
    loss='mean_squared_error',
    metrics=[
        "mean_squared_error", 
    ]
)
model_nn.fit(
    train_onehot, 
    train_y_onehot, 
    epochs=100,
    batch_size=300,
    callbacks=[tensorboard, checkpoint]
)


""" model blending"""
def weight_opt(oof_lgbm, oof_nn, train_y):
    weight_oof_lgbm = np.inf
    best_rmse = np.inf

    for i in np.arange(0, 1.01, 0.05):
        #mae_blend = np.zeros(oof1.shape[0])
        rmse_blend = np.sqrt(mean_squared_error(train_y, (i * oof_lgbm + (1-i) * oof_nn)))
        if rmse_blend < best_rmse:
            best_rmse = rmse_blend
            weight_oof_lgbm = round(i, 2)

        print(str(round(i, 2)) + ' :   (Blend) is ', round(best_rmse, 6))

    print('-'*36)
    print('Best weight for pred_lgbm: ', weight_oof_lgbm)
    print('Best weight for pred_nn: ', round(1-weight_oof_lgbm, 2))
    print('Best mean squared error (Blend): ', round(best_rmse, 6))

    return weight_oof_lgbm, round(1-weight_oof_lgbm, 2)

weight_lgbm, weight_nn = weight_opt(oof_lgbm, oof_nn, train_y_label)


""" predict"""
res_lgbm = model_lgbm.predict(test_label)
res_nn = model_nn.predict_on_batch(test_onehot)

res = weight_lgbm * res_lgbm + weight_nn * res_nn


""" export submit file"""
result = pd.DataFrame(test.index, columns=["id"])
result["y"] = res
result.to_csv("../output/result_realestate_20200426_02.csv", index=False)