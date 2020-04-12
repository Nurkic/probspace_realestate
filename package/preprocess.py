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


def convert_construction_year(df: pd.DataFrame) -> pd.DataFrame:
    """和暦を西暦に変換する
    '戦前'は昭和20年とした
    新たに追加される列名 -> 建築年(和暦), 年号, 和暦年数
    """
    df["建築年(和暦)"] = df["建築年"]
    df["建築年"].dropna(inplace=True)
    df["建築年"] = df["建築年"].str.replace("戦前", "昭和20年")
    df["年号"] = df["建築年"].str[:2]
    df["和暦年数"] = df["建築年"].str[2:].str.strip("年").fillna(0).astype(int)
    df.loc[df["年号"] == "昭和", "建築年"] = df["和暦年数"] + 1925
    df.loc[df["年号"] == "平成", "建築年"] = df["和暦年数"] + 1988
    return df


def to_onehot(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """pandas.DataFrameの要素をone-hotなベクトルに変換する
    変換したい列名をリストで渡す

    Parameters
    ----------
    columns : list
        列名のリスト
    """
    df[columns].dropna(inplace=True)
    tmp = df[columns].str.get_dummies("、")
    df = pd.concat([df, tmp], axis=1)
    # 冪等性を考慮して
    df = df.loc[:,~df.columns.duplicated()]
    return df


def to_label(df: pd.DataFrame, column: str, th: int = 100) -> pd.DataFrame:
    """ラベルエンコードするための前処理
    単語が組み合わさっているもののうち，出現回数が少ないものを1つにまとめる
    
    Parameters
    ----------
    th : int
        出現回数の閾値(default 100)
    """
    _df = df.copy()
    category_dict = _df[column].value_counts().to_dict()
    misc_list = [key for key, value in category_dict.items() if len(key.split("、")) == 2 or value <= th]
    _df[column] = _df[column].mask(_df[column].isin(misc_list), "misc")
    return _df


def direction_to_int(df: pd.DataFrame, column: str) -> pd.DataFrame:
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
    _df = df.copy()
    _df[column] = _df[column].map(DIRECTION_ANGLE_DICT)
    return _df
