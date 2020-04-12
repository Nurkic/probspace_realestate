import numpy as np
import pandas as pd

import json

import category_encoders as ce


class _Rename:
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df


    """ Convert train and test column names to English"""
    def _rename_t(self) -> pd.DataFrame:
        with open("names.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        df = self.df.rename(columns=d)

        return df


    """ Unify published column names"""
    def _rename_p(self) -> pd.DataFrame:
        pair = {"所在地コード":"市区町村コード", "建蔽率":"建ぺい率（％）", "容積率":"容積率（％）", "駅名":"最寄駅：名称", 
            "地積":"面積（㎡）", "市区町村名":"市区町村名", '前面道路の幅員':'前面道路：幅員（ｍ）', 
            "前面道路の方位区分":"前面道路：方位", "前面道路区分":"前面道路：種類", "形状区分":"土地の形状", 
            "用途区分":"都市計画"}

        df = self.df.rename(columns=pair)

        return df


class _Encoder:
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df



    """ label encoding"""
    def _cat_encoder(self) -> pd.DataFrame:
        object_cols = []
        for column in self.df.columns:
            if self.df[column].dtype == object:
                object_cols.append(column)
        
        ce_oe = ce.OrdinalEncoder(cols=object_cols,handle_unknown='impute')
        df = ce_oe.fit_transform(self.df)

        return df


    """ one hot encoding"""
    def _onehot_encoder(self) -> pd.DataFrame:
        df = pd.get_dummies(self.df, drop_first=True, dummy_na=False)
    
        return df

    
    """ Adjust the number of label types"""
    def _relabeler(
        self,
        colname: str,
        k: int = 100
        ) -> pd.DataFrame:
        count_df = self.df[colname].value_counts().rename_axis[colname].reset_index(name="counts")
        adj_list = list(count_df[colname][count_df["counts"] < k])
        self.df[colname] = self.df[colname].replace(adj_list, "misc")

        return self.df


class Preprocessor(_Rename, _Encoder):
    def __init__(self, df: pd.DataFrame):
        super(Preprocessor, self).__init__(df)

    def to_onehot(self) -> pd.DataFrame:
        """Convert a pandas.DataFrame element to a one-hot vector
        """
        tmp = self._onehot_encoder()
        df = pd.concat([self.df, tmp], axis=1)
        # for idempotent
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def to_label(self, column: str, th: int = 100) -> pd.DataFrame:
        """Preprocessing for label-encoding.
        Combine the fewest frequent combinations of words into one.
        
        Parameters
        ----------
        th : int
            threshold of the number of occurences (default 100)
        """
        df = self.df.copy()
        category_dict = df[column].value_counts().to_dict()
        misc_list = [key for key, value in category_dict.items() if len(key.split("、")) == 2 or value <= th]
        df[column] = df[column].mask(df[column].isin(misc_list), "misc")
        return df

    def convert_construction_year(self) -> pd.DataFrame:
        """和暦を西暦に変換する
        '戦前'は昭和20年とした
        新たに追加される列名 -> 建築年(和暦), 年号, 和暦年数
        """
        df = self.df.copy()
        df["建築年(和暦)"] = df["建築年"]
        df["建築年"].dropna(inplace=True)
        df["建築年"] = df["建築年"].str.replace("戦前", "昭和20年")
        df["年号"] = df["建築年"].str[:2]
        df["和暦年数"] = df["建築年"].str[2:].str.strip("年").fillna(0).astype(int)
        df.loc[df["年号"] == "昭和", "建築年"] = df["和暦年数"] + 1925
        df.loc[df["年号"] == "平成", "建築年"] = df["和暦年数"] + 1988
        return df

    def direction_to_int(self, column: str) -> pd.DataFrame:
        """方角を整数に変換する
        東を0として，北東を1，北を2...というふうに反時計回りに1ずつ増える
        接面道路無は-1
        整数に45をかけることで角度に変換できる
        """
        DIRECTION_ANGLE_DICT = {
            "東": 0,
            "北東": 1,
            "北": 2,
            "北西": 3,
            "西": 4,
            "南西": 5,
            "南": 6,
            "南東": 7,
            "接面道路無": -1
        }
        df = self.df.copy()
        df[column] = df[column].map(DIRECTION_ANGLE_DICT)
        return df

    def convert_trading_point(self) -> pd.DataFrame:
        def f(x: str):
            TABLE = {
                "１": 0,
                "２": 1,
                "３": 2,
                "４": 3
            }
            l = x.split("年第")
            return float(l[0]) + TABLE[l[1][0]] * 0.25

        df = self.df.copy()
        df["取引時点"] = df["取引時点"].map(f)
        return df

    def all(self):
        self.df = self.to_onehot()
        self.df = self.to_label("建物の構造", 100)
        self.df = self.to_label("用途", 100)
        self.df = self.convert_construction_year()
        self.df = self.direction_to_int("前面道路：方位")
        self.df = self.convert_trading_point()
        return self.df
