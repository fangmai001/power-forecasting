import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from models.xgboost_model import XGBoostModel
from models.prophet_model import ProphetModel
from features.feature_generator import FeatureGenerator
from features.future_features import FutureFeatureGenerator

class PowerPredictor:
    """電力預測器類別"""
    
    def __init__(self):
        self.xgb_model = XGBoostModel()
        self.prophet_model = ProphetModel()
        self.feature_generator = FeatureGenerator()
        self.future_generator = FutureFeatureGenerator()
        
    def prepare_data(self,
                    df: pd.DataFrame,
                    target_column: str = 'power_consumption') -> tuple:
        """
        準備訓練數據
        
        Args:
            df: 原始數據框
            target_column: 目標變數欄位名稱
            
        Returns:
            處理後的特徵數據框和目標變數
        """
        # 生成時間特徵
        df = self.feature_generator.generate_time_features(df)
        
        # 處理缺失值
        df = self.feature_generator.handle_missing_values(
            df,
            numeric_columns=[target_column]
        )
        
        # 分離特徵和目標變數
        feature_columns = [col for col in df.columns 
                         if col not in ['date', target_column]]
        
        return df[feature_columns], df[target_column]
    
    def train_models(self,
                    train_df: pd.DataFrame,
                    target_column: str = 'power_consumption') -> None:
        """
        訓練 XGBoost 和 Prophet 模型
        
        Args:
            train_df: 訓練數據框
            target_column: 目標變數欄位名稱
        """
        # 訓練 XGBoost
        X, y = self.prepare_data(train_df, target_column)
        self.xgb_model.train(X, y, X.columns.tolist())
        
        # 訓練 Prophet
        prophet_df = self.prophet_model.prepare_data(
            train_df,
            date_column='date',
            target_column=target_column
        )
        self.prophet_model.train(prophet_df)
        
    def predict(self,
               start_date: str,
               periods: int,
               freq: str = 'D') -> Dict[str, pd.DataFrame]:
        """
        使用兩個模型進行預測
        
        Args:
            start_date: 預測開始日期
            periods: 預測期數
            freq: 頻率（D=日，W=週，M=月）
            
        Returns:
            包含兩個模型預測結果的字典
        """
        # 生成未來特徵
        future_df = self.future_generator.generate_future_dates(
            start_date,
            periods,
            freq
        )
        
        # XGBoost 預測
        xgb_pred = self.xgb_model.predict(future_df)
        xgb_results = pd.DataFrame({
            'date': future_df['date'],
            'power_consumption': xgb_pred
        })
        
        # Prophet 預測
        prophet_forecast = self.prophet_model.predict(periods, freq)
        prophet_results = pd.DataFrame({
            'date': prophet_forecast['ds'],
            'power_consumption': prophet_forecast['yhat']
        })
        
        return {
            'xgboost': xgb_results,
            'prophet': prophet_results
        } 