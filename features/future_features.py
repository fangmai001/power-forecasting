import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from .feature_generator import FeatureGenerator

class FutureFeatureGenerator:
    """生成未來時間特徵的類別"""
    
    def __init__(self, feature_generator: Optional[FeatureGenerator] = None):
        self.feature_generator = feature_generator or FeatureGenerator()
    
    def generate_future_dates(self,
                            start_date: str,
                            periods: int,
                            freq: str = 'D') -> pd.DataFrame:
        """
        生成未來日期序列
        
        Args:
            start_date: 開始日期
            periods: 預測期數
            freq: 頻率（D=日，W=週，M=月）
            
        Returns:
            包含未來日期的數據框
        """
        dates = pd.date_range(
            start=start_date,
            periods=periods,
            freq=freq
        )
        
        future_df = pd.DataFrame({'date': dates})
        
        # 使用 FeatureGenerator 生成時間特徵
        future_df = self.feature_generator.generate_time_features(future_df)
        
        return future_df
    
    def align_features(self,
                      historical_df: pd.DataFrame,
                      future_df: pd.DataFrame) -> pd.DataFrame:
        """
        確保未來特徵與歷史特徵格式一致
        
        Args:
            historical_df: 歷史數據框
            future_df: 未來數據框
            
        Returns:
            對齊後的未來數據框
        """
        # 獲取歷史數據中的特徵欄位
        feature_columns = [col for col in historical_df.columns 
                         if col not in ['date', 'power_consumption']]
        
        # 確保未來數據框包含所有必要的特徵
        for col in feature_columns:
            if col not in future_df.columns:
                future_df[col] = 0
        
        # 按照歷史數據的欄位順序排列
        future_df = future_df[['date'] + feature_columns]
        
        return future_df 