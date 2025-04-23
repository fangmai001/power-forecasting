import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

class DataProcessor:
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """
        初始化資料處理器
        
        Args:
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        載入原始資料
        
        Args:
            file_path: 資料檔案路徑
            
        Returns:
            pd.DataFrame: 載入的資料
        """
        df = pd.read_csv(file_path)
        # 將 date 欄位重命名為 timestamp
        df = df.rename(columns={'date': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分割資料為訓練集、驗證集和測試集
        
        Args:
            df: 輸入資料
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 訓練集、驗證集、測試集
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train_data = df[:train_end]
        val_data = df[train_end:val_end]
        test_data = df[val_end:]
        
        return train_data, val_data, test_data
    
    def scale_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                  test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        標準化資料
        
        Args:
            train_data: 訓練集
            val_data: 驗證集
            test_data: 測試集
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]: 標準化後的資料和標準化參數
        """
        # 只對數值型特徵進行標準化
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        
        # 擬合訓練集
        self.scaler.fit(train_data[numeric_cols])
        
        # 轉換所有資料集
        train_scaled = train_data.copy()
        val_scaled = val_data.copy()
        test_scaled = test_data.copy()
        
        train_scaled[numeric_cols] = self.scaler.transform(train_data[numeric_cols])
        val_scaled[numeric_cols] = self.scaler.transform(val_data[numeric_cols])
        test_scaled[numeric_cols] = self.scaler.transform(test_data[numeric_cols])
        
        # 保存標準化參數
        scaler_params = {
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_,
            'feature_names': numeric_cols.tolist()
        }
        
        return train_scaled, val_scaled, test_scaled, scaler_params
    
    def process_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        完整的資料處理流程
        
        Args:
            file_path: 資料檔案路徑
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]: 處理後的資料和標準化參數
        """
        # 載入資料
        df = self.load_data(file_path)
        
        # 分割資料
        train_data, val_data, test_data = self.split_data(df)
        
        return train_data, val_data, test_data, None
        
        # # 標準化資料
        # train_scaled, val_scaled, test_scaled, scaler_params = self.scale_data(
        #     train_data, val_data, test_data
        # )
        
        # return train_scaled, val_scaled, test_scaled, scaler_params 