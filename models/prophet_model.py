import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple, Dict
from datetime import datetime

class ProphetModel:
    """Prophet 模型類別"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'seasonality_mode': 'multiplicative'
        }
        self.model = Prophet(**self.params)
        
    def prepare_data(self,
                    df: pd.DataFrame,
                    date_column: str = 'date',
                    target_column: str = 'power_consumption') -> pd.DataFrame:
        """
        準備 Prophet 所需的數據格式
        
        Args:
            df: 輸入數據框
            date_column: 日期欄位名稱
            target_column: 目標變數欄位名稱
            
        Returns:
            Prophet 格式的數據框
        """
        prophet_df = df[[date_column, target_column]].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df
    
    def train(self, df: pd.DataFrame) -> None:
        """
        訓練模型
        
        Args:
            df: Prophet 格式的數據框（含 ds 和 y 欄位）
        """
        self.model.fit(df)
        
    def predict(self, periods: int, freq: str = 'D') -> pd.DataFrame:
        """
        進行預測
        
        Args:
            periods: 預測期數
            freq: 頻率（D=日，W=週，M=月）
            
        Returns:
            預測結果數據框
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast
    
    def evaluate(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> Tuple[float, float, float]:
        """
        評估模型績效
        
        Args:
            y_true: 實際值
            y_pred: 預測值
            
        Returns:
            MAE, RMSE, MAPE
        """
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return mae, rmse, mape
    
    def get_components(self) -> Dict:
        """
        獲取預測的各個組成部分
        
        Returns:
            包含趨勢、季節性等組成部分的字典
        """
        return {
            'trend': self.model.trend,
            'yearly': self.model.yearly_seasonality,
            'weekly': self.model.weekly_seasonality,
            'daily': self.model.daily_seasonality
        } 