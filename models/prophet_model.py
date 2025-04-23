from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class ProphetModel:
    def __init__(self, params: Dict = None):
        """
        初始化 Prophet 模型
        
        Args:
            params: Prophet 模型參數
        """
        self.params = params or {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative',
            'changepoint_range': 0.8,
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True
        }
        self.model = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        準備 Prophet 所需的資料格式
        
        Args:
            df: 輸入資料
            target_col: 目標變數名稱
            
        Returns:
            pd.DataFrame: 格式化後的資料
        """
        prophet_df = df.reset_index()
        prophet_df = prophet_df.rename(columns={
            'timestamp': 'ds',
            target_col: 'y'
        })
        return prophet_df[['ds', 'y']]
    
    def train(self, df: pd.DataFrame, target_col: str):
        """
        訓練 Prophet 模型
        
        Args:
            df: 輸入資料
            target_col: 目標變數名稱
        """
        # 準備資料
        prophet_df = self.prepare_data(df, target_col)
        
        # 初始化模型
        self.model = Prophet(**self.params)
        
        # 訓練模型
        self.model.fit(prophet_df)
    
    def predict(self, df: pd.DataFrame, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        使用訓練好的模型進行預測
        
        Args:
            df: 輸入資料
            periods: 預測期數
            freq: 預測頻率
            
        Returns:
            pd.DataFrame: 預測結果
        """
        # 建立未來時間序列
        future = pd.DataFrame({
            'ds': df.index
        })
        
        # 進行預測
        forecast = self.model.predict(future)
        
        # 整理預測結果
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        result = result.rename(columns={'ds': 'timestamp'})
        result.set_index('timestamp', inplace=True)
        
        return result
    
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        評估模型表現
        
        Args:
            y_true: 真實值
            y_pred: 預測值
            
        Returns:
            Dict[str, float]: 評估指標
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }
        return metrics
    
    def get_components(self) -> Dict[str, pd.DataFrame]:
        """
        獲取模型組件
        
        Returns:
            Dict[str, pd.DataFrame]: 模型組件
        """
        if self.model is None:
            return None
        
        components = {
            'trend': self.model.history['trend'],
            'yearly': self.model.history['yearly'],
            'weekly': self.model.history['weekly'],
            'daily': self.model.history['daily']
        }
        return components
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 模型保存路徑
        """
        if self.model is not None:
            with open(path, 'wb') as f:
                import pickle
                pickle.dump(self.model, f)
    
    def load_model(self, path: str):
        """
        載入模型
        
        Args:
            path: 模型路徑
        """
        with open(path, 'rb') as f:
            import pickle
            self.model = pickle.load(f) 