import pandas as pd
import numpy as np
from typing import List

class FeatureGenerator:
    """特徵工程類別，用於生成時間相關特徵"""
    
    @staticmethod
    def generate_time_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        生成時間相關特徵
        
        Args:
            df: 包含日期欄位的數據框
            date_column: 日期欄位名稱
            
        Returns:
            添加時間特徵後的數據框
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # 基本時間特徵
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['week'] = df[date_column].dt.isocalendar().week
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # 週期性特徵
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['week_sin'] = np.sin(2 * np.pi * df['week']/52)
        df['week_cos'] = np.cos(2 * np.pi * df['week']/52)
        
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