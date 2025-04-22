import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelEvaluator:
    """模型評估器類別"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, float]:
        """
        計算評估指標
        
        Args:
            y_true: 實際值
            y_pred: 預測值
            
        Returns:
            包含各項指標的字典
        """
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def compare_models(self,
                      actual_df: pd.DataFrame,
                      predictions: Dict[str, pd.DataFrame],
                      target_column: str = 'power_consumption') -> Dict[str, Dict[str, float]]:
        """
        比較不同模型的預測績效
        
        Args:
            actual_df: 實際值數據框
            predictions: 各模型預測結果字典
            target_column: 目標變數欄位名稱
            
        Returns:
            各模型評估指標的字典
        """
        results = {}
        
        for model_name, pred_df in predictions.items():
            # 確保日期對齊
            merged = pd.merge(
                actual_df,
                pred_df,
                on='date',
                suffixes=('_actual', '_pred')
            )
            
            y_true = merged[f'{target_column}_actual']
            y_pred = merged[f'{target_column}_pred']
            
            results[model_name] = self.calculate_metrics(y_true, y_pred)
            
        return results
    
    def plot_predictions(self,
                        actual_df: pd.DataFrame,
                        predictions: Dict[str, pd.DataFrame],
                        target_column: str = 'power_consumption',
                        use_plotly: bool = True) -> None:
        """
        繪製預測結果比較圖
        
        Args:
            actual_df: 實際值數據框
            predictions: 各模型預測結果字典
            target_column: 目標變數欄位名稱
            use_plotly: 是否使用 Plotly（否則使用 Matplotlib）
        """
        if use_plotly:
            fig = go.Figure()
            
            # 添加實際值
            fig.add_trace(go.Scatter(
                x=actual_df['date'],
                y=actual_df[target_column],
                name='實際值',
                line=dict(color='black', width=2)
            ))
            
            # 添加各模型預測值
            colors = ['red', 'blue']
            for (model_name, pred_df), color in zip(predictions.items(), colors):
                fig.add_trace(go.Scatter(
                    x=pred_df['date'],
                    y=pred_df[target_column],
                    name=f'{model_name} 預測',
                    line=dict(color=color, width=2)
                ))
            
            fig.update_layout(
                title='用電量預測比較',
                xaxis_title='日期',
                yaxis_title='用電量',
                template='plotly_white'
            )
            fig.show()
            
        else:
            plt.figure(figsize=(12, 6))
            
            # 繪製實際值
            plt.plot(actual_df['date'],
                    actual_df[target_column],
                    'k-',
                    label='實際值',
                    linewidth=2)
            
            # 繪製各模型預測值
            colors = ['r-', 'b-']
            for (model_name, pred_df), color in zip(predictions.items(), colors):
                plt.plot(pred_df['date'],
                        pred_df[target_column],
                        color,
                        label=f'{model_name} 預測',
                        linewidth=2)
            
            plt.title('用電量預測比較')
            plt.xlabel('日期')
            plt.ylabel('用電量')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show() 