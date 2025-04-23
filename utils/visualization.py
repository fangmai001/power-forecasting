import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

class PowerVisualizer:
    """用電量視覺化工具類別"""
    
    @staticmethod
    def plot_consumption_trend(df: pd.DataFrame,
                             date_column: str = 'date',
                             target_column: str = 'power_consumption',
                             use_plotly: bool = True) -> go.Figure:
        """
        繪製用電量趨勢圖
        
        Args:
            df: 數據框
            date_column: 日期欄位名稱
            target_column: 目標變數欄位名稱
            use_plotly: 是否使用 Plotly
            
        Returns:
            Plotly 圖表物件
        """
        if use_plotly:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df[target_column],
                mode='lines',
                name='用電量'
            ))
            
            fig.update_layout(
                title='工廠用電量趨勢',
                xaxis_title='日期',
                yaxis_title='用電量 (kWh)',
                template='plotly_white'
            )
            return fig
            
        else:
            plt.figure(figsize=(12, 6))
            plt.plot(df[date_column], df[target_column], 'b-')
            plt.title('工廠用電量趨勢')
            plt.xlabel('日期')
            plt.ylabel('用電量 (kWh)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return plt.gcf()
    
    @staticmethod
    def plot_seasonal_patterns(df: pd.DataFrame,
                             date_column: str = 'date',
                             target_column: str = 'power_consumption') -> go.Figure:
        """
        繪製季節性模式圖
        
        Args:
            df: 數據框
            date_column: 日期欄位名稱
            target_column: 目標變數欄位名稱
            
        Returns:
            Plotly 圖表物件
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('月度模式', '週度模式', '日內模式', '週末vs平日')
        )
        
        # 月度模式
        monthly = df.groupby(df[date_column].dt.month)[target_column].mean()
        fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly.values, name='月度'),
            row=1, col=1
        )
        
        # 週度模式
        weekly = df.groupby(df[date_column].dt.dayofweek)[target_column].mean()
        fig.add_trace(
            go.Scatter(x=['一', '二', '三', '四', '五', '六', '日'],
                      y=weekly.values,
                      name='週度'),
            row=1, col=2
        )
        
        # 日內模式（如果有小時數據）
        if 'hour' in df.columns:
            hourly = df.groupby('hour')[target_column].mean()
            fig.add_trace(
                go.Scatter(x=hourly.index, y=hourly.values, name='日內'),
                row=2, col=1
            )
        
        # 週末vs平日
        weekend = df[date_column].dt.dayofweek.isin([5, 6])
        weekend_avg = df[weekend][target_column].mean()
        weekday_avg = df[~weekend][target_column].mean()
        fig.add_trace(
            go.Bar(x=['平日', '週末'],
                  y=[weekday_avg, weekend_avg],
                  name='週末比較'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text='用電量季節性模式分析',
            showlegend=False
        )
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame,
                              top_n: int = 10) -> plt.Figure:
        """
        繪製特徵重要性圖
        
        Args:
            importance_df: 特徵重要性數據框
            top_n: 顯示前 N 個重要特徵
            
        Returns:
            Matplotlib 圖表物件
        """
        plt.figure(figsize=(10, 6))
        
        # 選取前 N 個特徵
        top_features = importance_df.nlargest(top_n, 'importance')
        
        # 繪製條形圖
        sns.barplot(x='importance',
                   y='feature',
                   data=top_features,
                   palette='viridis')
        
        plt.title(f'前 {top_n} 個重要特徵')
        plt.xlabel('重要性分數')
        plt.ylabel('特徵名稱')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_error_distribution(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model_name: str = 'Model') -> plt.Figure:
        """
        繪製預測誤差分布圖
        
        Args:
            y_true: 實際值
            y_pred: 預測值
            model_name: 模型名稱
            
        Returns:
            Matplotlib 圖表物件
        """
        errors = y_true - y_pred
        
        plt.figure(figsize=(12, 5))
        
        # 子圖1：誤差直方圖
        plt.subplot(1, 2, 1)
        sns.histplot(errors, kde=True)
        plt.title(f'{model_name} 預測誤差分布')
        plt.xlabel('誤差')
        plt.ylabel('頻率')
        
        # 子圖2：實際值vs預測值散點圖
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'r--')
        plt.title('實際值 vs 預測值')
        plt.xlabel('實際值')
        plt.ylabel('預測值')
        
        plt.tight_layout()
        return plt.gcf() 