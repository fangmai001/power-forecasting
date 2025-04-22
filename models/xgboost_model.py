import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Tuple, Dict

class XGBoostModel:
    """XGBoost 模型類別"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.model = XGBRegressor(**self.params)
        
    def train(self,
             X: pd.DataFrame,
             y: pd.Series,
             feature_columns: List[str]) -> None:
        """
        訓練模型
        
        Args:
            X: 特徵數據框
            y: 目標變數
            feature_columns: 特徵欄位列表
        """
        self.feature_columns = feature_columns
        self.model.fit(X[feature_columns], y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        進行預測
        
        Args:
            X: 特徵數據框
            
        Returns:
            預測結果
        """
        return self.model.predict(X[self.feature_columns])
    
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
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return mae, rmse, mape
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        獲取特徵重要性
        
        Returns:
            特徵重要性數據框
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False) 