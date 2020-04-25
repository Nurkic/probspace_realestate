import numpy as np
import pandas as pd

import json
import re

import category_encoders as ce



class _Rename:
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df


    """ Convert train and test column names to English"""
    def rename_t(self) -> pd.DataFrame:
        with open("names.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        df = self.df.rename(columns=d)

        return df


    """ Unify published column names"""
    def rename_p(self) -> pd.DataFrame:
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
        object_cols = [
            'Type','Region','MunicipalityCode','Prefecture','Municipality','DistrictName','NearestStation',
            'FloorPlan','LandShape','Structure','Use','Purpose','Classification','CityPlanning',
            'Renovation','Remarks','L','D','K','S','R','Maisonette','OpenFloor','Studio','era_name'
            ]
        
        
        ce_oe = ce.OrdinalEncoder(cols=object_cols,handle_unknown='impute')
        df = ce_oe.fit_transform(self.df)

        for obj_col in object_cols:
            df[obj_col] = df[obj_col].astype("category")

        return df


    """ one hot encoding"""
    def _onehot_encoder(self, columns: list) -> pd.DataFrame:
        df = pd.get_dummies(self.df[columns], drop_first=True, dummy_na=False)
    
        return df

    
    """ Adjust the number of label types"""
    def relabeler(
        self,
        column: str,
        th: int = 100,
        comma_sep: bool = False
        ) -> pd.DataFrame:
        df = self.df.copy()
        category_dict = df[column].value_counts().to_dict()
        if comma_sep:
            misc_list = [key for key, value in category_dict.items() if len(key.split("、")) == 2 or value < th]
        else:
            misc_list = [key for key, value in category_dict.items() if value < th]
        df[column] = df[column].mask(df[column].isin(misc_list), "misc")

        return df

  
class Preprocessor(_Rename, _Encoder):
    def __init__(self, df: pd.DataFrame):
        super(Preprocessor, self).__init__(df)
        
    def to_onehot(self) -> pd.DataFrame:
        """Convert a pandas.DataFrame element to a one-hot vector
        """
        df = self.df.copy()
        cols = [
            'Type','Region','MunicipalityCode','Prefecture','Municipality','DistrictName','NearestStation',
            'FloorPlan','LandShape','Structure','Use','Purpose','Classification','CityPlanning',
            'Renovation','Remarks','era_name'
            ]
        tmp = self._onehot_encoder(cols)
        df = pd.concat([df, tmp], axis=1)
        # for idempotent
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def convert_construction_year(self) -> pd.DataFrame:
        """和暦を西暦に変換する
        '戦前'は昭和20年とした
        新たに追加される列名 -> 建築年(和暦), 年号, 和暦年数
        """
        df = self.df.copy()
        df["建築年"].dropna(inplace=True)
        df["建築年"] = df["建築年"].str.replace("戦前", "昭和20年")
        df["era_name"] = df["建築年"].str[:2]
        df["和暦年数"] = df["建築年"].str[2:].str.strip("年").fillna(0).astype(int)
        df.loc[df["era_name"] == "昭和", "建築年"] = df["和暦年数"] + 1925
        df.loc[df["era_name"] == "平成", "建築年"] = df["和暦年数"] + 1988
        df["建築年"] = pd.to_numeric(df["建築年"], errors="coerce")
        df = df.drop("和暦年数", axis=1)
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

    def floor(self) -> pd.DataFrame:
        df = self.df.copy()
        trans_table = str.maketrans({"１":"1", "２":"2", "３":"3", "４":"4","５":"5", "６":"6","７":"7", "８":"8"})
        df['間取り'] = df['間取り'].map(lambda x: x.translate(trans_table) if type(x) is str else x)
        df['NumberOfRooms'] = df['間取り'].map(
            lambda x: 1 if (type(x) == float) or (x == "メゾネット") or (x == "オープンフロア") or (x == "スタジオ") 
            else int(re.search('[0-9]+', x).group(0))
            )
        df['L'] = df['間取り'].map(lambda x: 1 if 'Ｌ' in str(x) else 0)
        df['D'] = df['間取り'].map(lambda x: 1 if 'Ｄ' in str(x) else 0)
        df['K'] = df['間取り'].map(lambda x: 1 if 'Ｋ' in str(x) else 0)
        df['S'] = df['間取り'].map(lambda x: 1 if 'Ｓ' in str(x) else 0)
        df['R'] = df['間取り'].map(lambda x: 1 if 'Ｒ' in str(x) else 0)
        df['Maisonette'] = df['間取り'].map(lambda x: 1 if 'メゾネット' in str(x) else 0)
        df['OpenFloor'] = df['間取り'].map(lambda x: 1 if 'オープンフロア' in str(x) else 0)
        df['Studio'] = df['間取り'].map(lambda x: 1 if 'スタジオ' in str(x) else 0)
    
        return df

    def min_from_sta(self) -> pd.DataFrame:
        TABLE = {
            "30分?60分": "45",
            "1H?1H30": "75",
            "1H30?2H": "105",
            "2H?": "120"
        }
        df = self.df.copy()
        df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace(TABLE)
        df["最寄駅：距離（分）"] = pd.to_numeric(df["最寄駅：距離（分）"], errors="coerce")

        return df

    def total_floor_area(self) -> pd.DataFrame:
        TABLE = {
            "2000㎡以上": 2000,
            "10m^2未満": 10
        }
        df = self.df.copy()
        df["延床面積（㎡）"] = df["延床面積（㎡）"].replace(TABLE)
        df["延床面積（㎡）"] = pd.to_numeric(df["延床面積（㎡）"], errors="coerce")
    
        return df

    def obj_to_numeric(self, cols: list):
        df = self.df.copy()
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        return df

    def building_age(self):
        df = self.df.copy()
        df["BuildingAge"] = df["取引時点"] - df["建築年"]

    def all(self, policy: str):
        self.df = self.floor()
        self.df = self.min_from_sta()
        self.df = self.total_floor_area()
        self.df = self.convert_construction_year()
        self.df = self.direction_to_int("前面道路：方位")
        self.df = self.convert_trading_point()
        self.df = self.relabeler("建物の構造", 100, True)
        self.df = self.relabeler("用途", 100, True)
        self.df = self.relabeler("市区町村名", 2000)
        self.df = self.obj_to_numeric(["面積（㎡）", "間口"])
        self.df = self.building_age()
        self.df = self.rename_t()
        if policy == "onehot":
            self.df = self.to_onehot()
        elif policy == "label":
            self.df = self._cat_encoder()
        else:
            raise ValueError('Select "onehot" or "label"')
        return self.df
