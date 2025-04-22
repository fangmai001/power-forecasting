import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加專案根目錄到系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecast.predictor import PowerPredictor
from forecast.evaluator import ModelEvaluator
from utils.visualization import PowerVisualizer

class Dashboard:
    """Streamlit 互動式儀表板類別"""
    
    def __init__(self):
        self.predictor = PowerPredictor()
        self.evaluator = ModelEvaluator()
        self.visualizer = PowerVisualizer()
        
    def load_data(self):
        """載入數據"""
        uploaded_file = st.file_uploader(
            "上傳歷史用電量數據 (CSV)",
            type=['csv']
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def run(self):
        """運行儀表板"""
        st.title('🔌 工廠用電預測系統')
        st.write('本系統使用 XGBoost 和 Prophet 模型預測工廠用電量')
        
        # 側邊欄設置
        st.sidebar.header('參數設置')
        
        # 載入數據
        df = self.load_data()
        
        if df is not None:
            # 顯示原始數據
            st.subheader('📊 歷史用電數據')
            st.write(df.head())
            
            # 繪製趨勢圖
            st.subheader('📈 用電量趨勢')
            self.visualizer.plot_consumption_trend(df, use_plotly=True)
            
            # 模型訓練
            if st.sidebar.button('訓練模型'):
                with st.spinner('模型訓練中...'):
                    self.predictor.train_models(df)
                st.success('模型訓練完成！')
            
            # 預測設置
            st.sidebar.subheader('預測設置')
            pred_periods = st.sidebar.slider('預測期數', 7, 365, 30)
            freq = st.sidebar.selectbox(
                '預測頻率',
                ['D', 'W', 'M'],
                format_func=lambda x: {
                    'D': '每日',
                    'W': '每週',
                    'M': '每月'
                }[x]
            )
            
            # 進行預測
            if st.sidebar.button('開始預測'):
                with st.spinner('預測計算中...'):
                    start_date = df['date'].max() + timedelta(days=1)
                    predictions = self.predictor.predict(
                        start_date.strftime('%Y-%m-%d'),
                        pred_periods,
                        freq
                    )
                    
                    # 顯示預測結果
                    st.subheader('🎯 預測結果')
                    tabs = st.tabs(['XGBoost', 'Prophet'])
                    
                    for tab, (model_name, pred_df) in zip(tabs, predictions.items()):
                        with tab:
                            st.write(f'{model_name} 模型預測結果：')
                            st.write(pred_df)
                            
                            # 下載預測結果
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                f'下載 {model_name} 預測結果',
                                csv,
                                f'{model_name.lower()}_predictions.csv',
                                'text/csv'
                            )
                    
                    # 繪製預測比較圖
                    st.subheader('📊 預測比較')
                    self.evaluator.plot_predictions(
                        df,
                        predictions,
                        use_plotly=True
                    )
            
            # 季節性分析
            if st.sidebar.checkbox('顯示季節性分析'):
                st.subheader('🌊 季節性模式分析')
                self.visualizer.plot_seasonal_patterns(df)
        
        else:
            st.info('請上傳 CSV 檔案開始分析')

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.run() 