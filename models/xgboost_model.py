import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class XGBoostModel:
    def __init__(self, params: Dict = None):
        """
        初始化 XGBoost 模型
        
        Args:
            params: XGBoost 模型參數
        """
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0
        }
        self.model = None
        self.feature_importance = None
        self.evals_result = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              early_stopping_rounds: int = 10) -> Dict:
        """
        訓練 XGBoost 模型
        
        Args:
            X_train: 訓練集特徵
            y_train: 訓練集目標
            X_val: 驗證集特徵
            y_val: 驗證集目標
            early_stopping_rounds: 早停輪數
            
        Returns:
            Dict: 訓練歷史
        """
        # 轉換為 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 訓練模型
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        self.evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=self.evals_result,
            verbose_eval=False
        )
        
        # 計算特徵重要性
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        return self.evals_result
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用訓練好的模型進行預測
        
        Args:
            X: 輸入特徵
            
        Returns:
            np.ndarray: 預測結果
        """
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        獲取特徵重要性
        
        Returns:
            pd.DataFrame: 特徵重要性排序
        """
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values())
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 模型保存路徑
        """
        self.model.save_model(path)
    
    def load_model(self, path: str):
        """
        載入模型
        
        Args:
            path: 模型路徑
        """
        self.model = xgb.Booster()
        self.model.load_model(path) 