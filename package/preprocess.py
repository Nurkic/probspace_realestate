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


    # Convert train and test column names to English
    def _rename_t(self) -> pd.DataFrame:
        with open("names.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        df = self.df.rename(columns=d)

        return df


    # Unify published column names
    def _rename_p(self) -> pd.DataFrame:
        pair = {"所在地コード":"市区町村コード", "建蔽率":"建ぺい率（％）", "容積率":"容積率（％）", "駅名":"最寄駅：名称", 
            "地積":"面積（㎡）", "市区町村名":"市区町村名", '前面道路の幅員':'前面道路：幅員（ｍ）', 
            "前面道路の方位区分":"前面道路：方位", "前面道路区分":"前面道路：種類", "形状区分":"土地の形状", 
            "用途区分":"都市計画"}

        df = self.df.rename(columns=pair)

        return df


class _CategoryEncoder:
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df

    def _cat_encoder(self) -> pd.DataFrame:
        object_cols = []
        for column in self.df.columns:
            if self.df[column].dtype == object:
                object_cols.append(column)
        
        ce_oe = ce.OrdinalEncoder(cols=object_cols,handle_unknown='impute')
        df = ce_oe.fit_transform(self.df)

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


def building_structure_to_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """建物の構造をone-hotなベクトルに変換する
    """
    df["建物の構造"].dropna(inplace=True)
    tmp = df["建物の構造"].str.get_dummies("、")
    df = pd.concat([df, tmp], axis=1)
    # 冪等性を考慮して
    df = df.loc[:,~df.columns.duplicated()]
    return df
