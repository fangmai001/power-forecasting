import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        """初始化評估器"""
        pass
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        計算評估指標
        
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
    
    def compare_models(self, results: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        比較不同模型的表現
        
        Args:
            results: 模型預測結果
            
        Returns:
            pd.DataFrame: 模型比較結果
        """
        comparison = []
        
        for model_name, model_results in results.items():
            metrics = self.calculate_metrics(
                model_results['y_true'],
                model_results['y_pred']
            )
            metrics['model'] = model_name
            comparison.append(metrics)
        
        return pd.DataFrame(comparison)
    
    def plot_predictions(self, results: Dict[str, Dict[str, pd.DataFrame]], 
                        title: str = 'Model Predictions Comparison'):
        """
        繪製預測結果比較圖
        
        Args:
            results: 模型預測結果
            title: 圖表標題
        """
        plt.figure(figsize=(12, 6))
        
        for model_name, model_results in results.items():
            plt.plot(model_results['y_pred'].index, 
                    model_results['y_pred'].values,
                    label=f'{model_name} Prediction')
        
        plt.plot(results[list(results.keys())[0]]['y_true'].index,
                results[list(results.keys())[0]]['y_true'].values,
                label='Actual',
                color='black',
                linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_residuals(self, results: Dict[str, Dict[str, pd.DataFrame]]):
        """
        繪製殘差圖
        
        Args:
            results: 模型預測結果
        """
        n_models = len(results)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for (model_name, model_results), ax in zip(results.items(), axes):
            residuals = model_results['y_true'] - model_results['y_pred']
            
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title(f'{model_name} Residuals Distribution')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 10):
        """
        繪製特徵重要性圖
        
        Args:
            feature_importance: 特徵重要性數據
            top_n: 顯示前 N 個重要特徵
        """
        plt.figure(figsize=(10, 6))
        
        # 選取前 N 個重要特徵
        top_features = feature_importance.head(top_n)
        
        # 繪製水平條形圖
        sns.barplot(x='importance', y='feature', data=top_features)
        
        plt.title('Top Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_forecast_intervals(self, forecast: pd.DataFrame, 
                              actual: pd.Series = None):
        """
        繪製預測區間圖
        
        Args:
            forecast: 預測結果（包含預測值和上下界）
            actual: 實際值（可選）
        """
        plt.figure(figsize=(12, 6))
        
        # 繪製預測區間
        plt.fill_between(forecast.index,
                        forecast['lower_bound'],
                        forecast['upper_bound'],
                        alpha=0.2,
                        label='Prediction Interval')
        
        # 繪製預測值
        plt.plot(forecast.index,
                forecast['prediction'],
                label='Prediction',
                color='blue')
        
        # 如果有實際值，則繪製
        if actual is not None:
            plt.plot(actual.index,
                    actual.values,
                    label='Actual',
                    color='black',
                    linestyle='--')
        
        plt.title('Forecast with Prediction Intervals')
        plt.xlabel('Date')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf() 