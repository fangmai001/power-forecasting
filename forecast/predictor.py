import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from models.xgboost_model import XGBoostModel
from models.prophet_model import ProphetModel
from features.feature_generator import FeatureGenerator

class PowerPredictor:
    def __init__(self, target_col: str = 'power_consumption'):
        """
        初始化預測器
        
        Args:
            target_col: 目標變數名稱
        """
        self.target_col = target_col
        self.feature_generator = FeatureGenerator()
        self.xgb_model = XGBoostModel()
        self.prophet_model = ProphetModel()
        self.scaler_params = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        準備特徵
        
        Args:
            df: 輸入資料
            
        Returns:
            pd.DataFrame: 加入特徵後的資料
        """
        return self.feature_generator.generate_all_features(df, self.target_col)
    
    def train_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """
        訓練模型
        
        Args:
            train_data: 訓練集
            val_data: 驗證集
        """
        print('train_data', train_data)
        print('val_data', val_data)
        # 準備特徵
        train_features = self.prepare_features(train_data)
        val_features = self.prepare_features(val_data)
        
        # 訓練 XGBoost 模型
        self.xgb_model.train(
            train_features.drop(columns=[self.target_col]),
            train_features[self.target_col],
            val_features.drop(columns=[self.target_col]),
            val_features[self.target_col]
        )
        
        # 訓練 Prophet 模型
        # 合併訓練集和驗證集
        combined_data = pd.concat([train_data, val_data])
        self.prophet_model.train(combined_data, self.target_col)
    
    def predict(self, df: pd.DataFrame, periods: int = 30, freq: str = 'D') -> Dict[str, pd.DataFrame]:
        """
        進行預測
        
        Args:
            df: 輸入資料
            periods: 預測期數
            freq: 預測頻率
            
        Returns:
            Dict[str, pd.DataFrame]: 預測結果
        """
        # 準備特徵
        features = self.prepare_features(df)
        
        # XGBoost 預測
        xgb_pred = self.xgb_model.predict(features.drop(columns=[self.target_col]))
        
        # Prophet 預測
        prophet_pred = self.prophet_model.predict(df, periods, freq)
        
        # 整合預測結果
        results = {
            'xgb': pd.DataFrame({
                'timestamp': df.index,
                'prediction': xgb_pred
            }).set_index('timestamp'),
            'prophet': prophet_pred
        }
        
        return results
    
    def ensemble_predict(self, df: pd.DataFrame, periods: int = 30, freq: str = 'D',
                        weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        整合預測
        
        Args:
            df: 輸入資料
            periods: 預測期數
            freq: 預測頻率
            weights: 模型權重
            
        Returns:
            pd.DataFrame: 整合後的預測結果
        """
        if weights is None:
            weights = {'xgb': 0.5, 'prophet': 0.5}
        
        # 取得各模型預測結果
        predictions = self.predict(df, periods, freq)
        
        # 整合預測
        ensemble_pred = (
            weights['xgb'] * predictions['xgb']['prediction'] +
            weights['prophet'] * predictions['prophet']['yhat']
        )
        
        # 計算預測區間
        lower_bound = (
            weights['xgb'] * predictions['xgb']['prediction'] +
            weights['prophet'] * predictions['prophet']['yhat_lower']
        )
        
        upper_bound = (
            weights['xgb'] * predictions['xgb']['prediction'] +
            weights['prophet'] * predictions['prophet']['yhat_upper']
        )
        
        # 整理結果
        result = pd.DataFrame({
            'prediction': ensemble_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        return result
    
    def save_models(self, xgb_path: str, prophet_path: str):
        """
        保存模型
        
        Args:
            xgb_path: XGBoost 模型路徑
            prophet_path: Prophet 模型路徑
        """
        self.xgb_model.save_model(xgb_path)
        self.prophet_model.save_model(prophet_path)
    
    def load_models(self, xgb_path: str, prophet_path: str):
        """
        載入模型
        
        Args:
            xgb_path: XGBoost 模型路徑
            prophet_path: Prophet 模型路徑
        """
        self.xgb_model.load_model(xgb_path)
        self.prophet_model.load_model(prophet_path) 