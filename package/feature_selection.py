import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor



""" cross validation"""
def cross_validator(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame
) -> float:
    scores = []

    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    for tr_idx, va_idx in kf.split(train_X):
        tr_X, va_X = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        
        reg = LGBMRegressor(class_weight=None,
        importance_type='split', learning_rate=0.1,
        n_iter_no_change=None, n_jobs=1,
        objective=None, param_distributions=None, random_state=71,
        study=None, timeout=None)
        reg.fit(tr_X, tr_y)
        va_pred = reg.predict(va_X)
        #print(len(va_pred))
        #print(len(va_y))
        score = np.sqrt(mean_squared_error(va_y, va_pred))
        scores.append(score)
    return np.mean(scores)

class FeatureSelector:
    def __init__(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame
    ) -> None:
        self.train_X = train_X
        self.train_y = train_y


    """ Greedy Forward Selection"""
    def greedy_forward_selection(
        self
    ) -> set:
        

        best_score = 99999.0
        candidates = np.random.RandomState(71).permutation(self.train_X.columns)
        selected = set([])

        print("start simple selection")
        for feature in candidates:
            fs = list(selected) + [feature]
            score = cross_validator(self.train_X[fs], self.train_y)
            
            if score < best_score:
                selected.add(feature)
                best_score = score
                #print(f'selected: {feature}')
                #print(f'score: {score}')

        print(f'selected features: {selected}')

        return selected


    """ target encoding"""
    def target_encodeing(
        self,
        test: pd.DataFrame
        ) -> pd.DataFrame:
        column_list = []
        
        train_X = self.train_X
        for column in train_X.columns:
            if train_X[column].dtype == category:
                column_list.append(column)
        for c in column_list:
            data_tmp = pd.DataFrame({c: train_X[c], "target": self.train_y})
            target_mean = data_tmp.groupby(c)["target"].mean()
            #print(target_mean)
            test[c] = test[c].map(target_mean)

            tmp = np.repeat(np.nan, train_X.shape[0])
            kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
            for idx_1, idx_2 in kf_encoding.split(train_X):
                target_mean = data_tmp.iloc[idx_1].groupby(c)["target"].mean()

                tmp[idx_2] = train_X[c].iloc[idx_2].map(target_mean)

            train_X[c] = tmp

        return train_X, test