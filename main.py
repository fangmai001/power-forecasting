import argparse
import pandas as pd
import numpy as np
from data.data_processor import DataProcessor
from forecast.predictor import PowerPredictor
from forecast.evaluator import ModelEvaluator
import matplotlib.pyplot as plt
import os

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='工廠用電預測系統')
    
    parser.add_argument('--data', type=str, required=True,
                      help='輸入數據路徑')
    parser.add_argument('--output', type=str, default='data/processed',
                      help='輸出目錄')
    parser.add_argument('--periods', type=int, default=30,
                      help='預測期數')
    parser.add_argument('--freq', type=str, default='D',
                      choices=['D', 'W', 'M'],
                      help='預測頻率（D=日，W=週，M=月）')
    parser.add_argument('--plot', action='store_true',
                      help='是否顯示圖表')
    parser.add_argument('--target_col', type=str, default='power_consumption',
                      help='目標變數名稱')
    
    return parser.parse_args()

def main():
    """主程式"""
    # 解析參數
    args = parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化資料處理器
    data_processor = DataProcessor()
    
    # 處理資料
    train_data, val_data, test_data, scaler_params = data_processor.process_data(args.data)
    
    # 初始化預測器
    predictor = PowerPredictor(target_col=args.target_col)
    
    # 訓練模型
    predictor.train_models(train_data, val_data)
    
    # 進行預測
    predictions = predictor.predict(test_data, args.periods, args.freq)
    
    # 整合預測
    ensemble_pred = predictor.ensemble_predict(test_data, args.periods, args.freq)
    
    # 初始化評估器
    evaluator = ModelEvaluator()
    
    # 準備評估結果
    results = {
        'xgb': {
            'y_true': test_data[args.target_col],
            'y_pred': predictions['xgb']['prediction']
        },
        'prophet': {
            'y_true': test_data[args.target_col],
            'y_pred': predictions['prophet']['yhat']
        }
    }
    
    # 比較模型表現
    comparison = evaluator.compare_models(results)
    print("\n模型比較結果：")
    print(comparison)
    
    # 保存預測結果
    ensemble_pred.to_csv(os.path.join(args.output, 'ensemble_predictions.csv'))
    comparison.to_csv(os.path.join(args.output, 'model_comparison.csv'))
    
    # 保存模型
    predictor.save_models(
        os.path.join(args.output, 'xgb_model.json'),
        os.path.join(args.output, 'prophet_model.pkl')
    )
    
    # 繪製圖表
    if args.plot:
        # 預測比較圖
        fig1 = evaluator.plot_predictions(results)
        fig1.savefig(os.path.join(args.output, 'predictions_comparison.png'))
        
        # 殘差圖
        fig2 = evaluator.plot_residuals(results)
        fig2.savefig(os.path.join(args.output, 'residuals.png'))
        
        # 預測區間圖
        fig3 = evaluator.plot_forecast_intervals(ensemble_pred, test_data[args.target_col])
        fig3.savefig(os.path.join(args.output, 'forecast_intervals.png'))
        
        # 特徵重要性圖
        feature_importance = predictor.xgb_model.get_feature_importance()
        if feature_importance is not None:
            fig4 = evaluator.plot_feature_importance(feature_importance)
            fig4.savefig(os.path.join(args.output, 'feature_importance.png'))
        
        plt.close('all')

if __name__ == '__main__':
    main() 