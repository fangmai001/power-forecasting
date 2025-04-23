import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureGenerator:
    """特徵工程類別，用於生成時間相關特徵"""
    
    def __init__(self):
        """初始化特徵生成器"""
        pass
    
    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成時間相關特徵
        
        Args:
            df: 輸入資料
            
        Returns:
            pd.DataFrame: 加入時間特徵後的資料
        """
        df = df.copy()
        
        # 基本時間特徵
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['dayofyear'] = df.index.dayofyear
        df['weekofyear'] = df.index.isocalendar().week
        
        # 週末標記
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        return df
    
    def generate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成週期性特徵
        
        Args:
            df: 輸入資料
            
        Returns:
            pd.DataFrame: 加入週期性特徵後的資料
        """
        df = df.copy()
        
        # 月份週期性特徵
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # 日期週期性特徵
        df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
        
        # 星期週期性特徵
        df['week_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['week_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        
        return df
    
    def generate_lag_features(self, df: pd.DataFrame, target_col: str, 
                            lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        生成滯後特徵
        
        Args:
            df: 輸入資料
            target_col: 目標變數名稱
            lags: 滯後期數列表
            
        Returns:
            pd.DataFrame: 加入滯後特徵後的資料
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        return df
    
    def generate_rolling_features(self, df: pd.DataFrame, target_col: str,
                                windows: List[int] = [7, 30]) -> pd.DataFrame:
        """
        生成滾動統計特徵
        
        Args:
            df: 輸入資料
            target_col: 目標變數名稱
            windows: 滾動視窗大小列表
            
        Returns:
            pd.DataFrame: 加入滾動特徵後的資料
        """
        df = df.copy()
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
            
        return df
    
    def generate_all_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        生成所有特徵
        
        Args:
            df: 輸入資料
            target_col: 目標變數名稱
            
        Returns:
            pd.DataFrame: 加入所有特徵後的資料
        """
        df = self.generate_time_features(df)
        df = self.generate_cyclical_features(df)
        df = self.generate_lag_features(df, target_col)
        df = self.generate_rolling_features(df, target_col)
        
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, 
                            numeric_columns: List[str],
                            method: str = 'linear') -> pd.DataFrame:
        """
        處理缺失值
        
        Args:
            df: 數據框
            numeric_columns: 需要處理缺失值的數值欄位
            method: 插值方法，預設為線性插值
            
        Returns:
            處理後的數據框
        """
        df = df.copy()
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method=method)
        return df 