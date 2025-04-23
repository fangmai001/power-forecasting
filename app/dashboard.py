import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from data.data_processor import DataProcessor
from forecast.predictor import PowerPredictor
from forecast.evaluator import ModelEvaluator
from typing import Dict, Tuple, Optional

class DashboardConfig:
    """儀表板配置類別"""
    def __init__(self):
        self.page_title = "工廠用電預測系統"
        self.page_icon = "🔌"
        self.layout = "wide"
        self.tabs = ["資料分析", "模型訓練", "預測結果"]

class DataManager:
    """資料管理類別"""
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """載入資料"""
        return pd.read_csv(file_path)
    
    @staticmethod
    def process_data(uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """處理資料"""
        data_processor = DataProcessor()
        return data_processor.process_data(uploaded_file)

class Visualization:
    """視覺化元件類別"""
    @staticmethod
    def plot_time_series(df: pd.DataFrame) -> go.Figure:
        """繪製時間序列圖"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['power_consumption'],
            mode='lines',
            name='用電量'
        ))
        fig.update_layout(
            title='用電量趨勢',
            xaxis_title='日期',
            yaxis_title='用電量',
            template='plotly_white'
        )
        return fig
    
    @staticmethod
    def plot_seasonality(df: pd.DataFrame) -> go.Figure:
        """繪製季節性分析圖"""
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        monthly_avg = df.groupby('month')['power_consumption'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_avg.index,
            y=monthly_avg.values,
            name='月平均用電量'
        ))
        fig.update_layout(
            title='月平均用電量',
            xaxis_title='月份',
            yaxis_title='用電量',
            template='plotly_white'
        )
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_importance: pd.DataFrame) -> go.Figure:
        """繪製特徵重要性圖"""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h'
        ))
        fig.update_layout(
            title='特徵重要性',
            xaxis_title='重要性',
            yaxis_title='特徵',
            template='plotly_white'
        )
        return fig
    
    @staticmethod
    def plot_forecast(test_data: pd.DataFrame, ensemble_pred: pd.DataFrame) -> go.Figure:
        """繪製預測結果圖"""
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # 實際值
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=test_data['power_consumption'],
                name="實際值",
                line=dict(color="black")
            ),
            secondary_y=False
        )
        
        # 預測值
        fig.add_trace(
            go.Scatter(
                x=ensemble_pred.index,
                y=ensemble_pred['prediction'],
                name="預測值",
                line=dict(color="blue")
            ),
            secondary_y=False
        )
        
        # 預測區間
        fig.add_trace(
            go.Scatter(
                x=ensemble_pred.index,
                y=ensemble_pred['upper_bound'],
                name="上界",
                line=dict(color="rgba(0,0,255,0.2)"),
                showlegend=False
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=ensemble_pred.index,
                y=ensemble_pred['lower_bound'],
                name="下界",
                fill="tonexty",
                line=dict(color="rgba(0,0,255,0.2)"),
                showlegend=False
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            title="用電量預測",
            xaxis_title="日期",
            yaxis_title="用電量",
            template="plotly_white"
        )
        return fig

class ModelManager:
    """模型管理類別"""
    def __init__(self):
        self.predictor = PowerPredictor()
        self.evaluator = ModelEvaluator()
    
    def train_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """訓練模型"""
        self.predictor.train_models(train_data, val_data)
    
    def predict(self, test_data: pd.DataFrame, periods: int, freq: str) -> Dict:
        """進行預測"""
        return self.predictor.predict(test_data, periods, freq)
    
    def ensemble_predict(self, test_data: pd.DataFrame, periods: int, freq: str, 
                        weights: Dict[str, float]) -> pd.DataFrame:
        """整合預測"""
        return self.predictor.ensemble_predict(test_data, periods, freq, weights)
    
    def evaluate_models(self, test_data: pd.DataFrame, predictions: Dict) -> pd.DataFrame:
        """評估模型"""
        results = {
            'xgb': {
                'y_true': test_data['power_consumption'],
                'y_pred': predictions['xgb']['prediction']
            },
            'prophet': {
                'y_true': test_data['power_consumption'],
                'y_pred': predictions['prophet']['yhat']
            }
        }
        return self.evaluator.compare_models(results)

def setup_sidebar() -> Tuple[int, str, float, float]:
    """設定側邊欄"""
    st.sidebar.header("設定")
    
    # 預測設定
    st.sidebar.subheader("預測設定")
    periods = st.sidebar.slider("預測期數", 1, 365, 30)
    freq = st.sidebar.selectbox("預測頻率", ['D', 'W', 'M'])
    
    # 模型權重
    st.sidebar.subheader("模型權重")
    xgb_weight = st.sidebar.slider("XGBoost 權重", 0.0, 1.0, 0.5)
    prophet_weight = st.sidebar.slider("Prophet 權重", 0.0, 1.0, 0.5)
    
    return periods, freq, xgb_weight, prophet_weight

def render_data_analysis_tab(df: pd.DataFrame) -> None:
    """渲染資料分析分頁"""
    st.header("資料分析")
    
    # 資料概覽
    st.subheader("資料概覽")
    st.write(df.describe())
    
    # 時間序列圖
    st.subheader("用電量趨勢")
    fig = Visualization.plot_time_series(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # 季節性分析
    st.subheader("季節性分析")
    fig = Visualization.plot_seasonality(df)
    st.plotly_chart(fig, use_container_width=True)

def render_model_training_tab(model_manager: ModelManager, train_data: pd.DataFrame, 
                            val_data: pd.DataFrame, test_data: pd.DataFrame, 
                            periods: int, freq: str) -> None:
    """渲染模型訓練分頁"""
    st.header("模型訓練")
    
    if st.button("開始訓練"):
        with st.spinner("訓練模型中..."):
            # 訓練模型
            model_manager.train_models(train_data, val_data)
            
            # 評估模型
            predictions = model_manager.predict(test_data, periods, freq)
            comparison = model_manager.evaluate_models(test_data, predictions)
            
            # 顯示評估結果
            st.subheader("模型比較")
            st.dataframe(comparison)
            
            # 顯示特徵重要性
            feature_importance = model_manager.predictor.xgb_model.get_feature_importance()
            if feature_importance is not None:
                st.subheader("特徵重要性")
                fig = Visualization.plot_feature_importance(feature_importance)
                st.plotly_chart(fig, use_container_width=True)

def render_forecast_tab(model_manager: ModelManager, test_data: pd.DataFrame, 
                       periods: int, freq: str, weights: Dict[str, float]) -> None:
    """渲染預測結果分頁"""
    st.header("預測結果")
    
    if st.button("開始預測"):
        with st.spinner("進行預測..."):
            # 進行預測
            predictions = model_manager.predict(test_data, periods, freq)
            ensemble_pred = model_manager.ensemble_predict(test_data, periods, freq, weights)
            
            # 顯示預測結果
            st.subheader("預測結果")
            fig = Visualization.plot_forecast(test_data, ensemble_pred)
            st.plotly_chart(fig, use_container_width=True)
            
            # 下載預測結果
            st.download_button(
                label="下載預測結果",
                data=ensemble_pred.to_csv().encode('utf-8'),
                file_name="predictions.csv",
                mime="text/csv"
            )

def main():
    """主程式"""
    # 初始化配置
    config = DashboardConfig()
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout
    )
    
    st.title(f"{config.page_icon} {config.page_title}")
    
    # 上傳資料
    uploaded_file = st.sidebar.file_uploader("上傳資料檔案", type=['csv'])
    
    if uploaded_file is not None:
        # 載入和處理資料
        df = DataManager.load_data(uploaded_file)
        train_data, val_data, test_data, scaler_params = DataManager.process_data(uploaded_file)
        
        # 設定側邊欄
        periods, freq, xgb_weight, prophet_weight = setup_sidebar()
        
        # 確保權重總和為 1
        if xgb_weight + prophet_weight != 1.0:
            st.sidebar.warning("模型權重總和必須為 1.0")
            return
        
        # 初始化模型管理器
        model_manager = ModelManager()
        
        # 主畫面分頁
        tab1, tab2, tab3 = st.tabs(config.tabs)
        
        with tab1:
            render_data_analysis_tab(df)
        
        with tab2:
            render_model_training_tab(model_manager, train_data, val_data, test_data, periods, freq)
        
        with tab3:
            weights = {'xgb': xgb_weight, 'prophet': prophet_weight}
            render_forecast_tab(model_manager, test_data, periods, freq, weights)
    else:
        st.info("請上傳資料檔案以開始分析")

if __name__ == "__main__":
    main() 