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

