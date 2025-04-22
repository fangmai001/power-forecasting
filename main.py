import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from forecast.predictor import PowerPredictor
from forecast.evaluator import ModelEvaluator
from utils.visualization import PowerVisualizer

def load_data(file_path: str) -> pd.DataFrame:
    """
    載入數據
    
    Args:
        file_path: CSV 檔案路徑
        
    Returns:
        數據框
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def save_predictions(predictions: dict, output_dir: str) -> None:
    """
    儲存預測結果
    
    Args:
        predictions: 預測結果字典
        output_dir: 輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, pred_df in predictions.items():
        file_path = output_path / f'{model_name.lower()}_predictions.csv'
        pred_df.to_csv(file_path, index=False)
        print(f'預測結果已儲存至：{file_path}')

def main():
    parser = argparse.ArgumentParser(description='工廠用電預測系統')
    parser.add_argument('--data', type=str, required=True,
                       help='輸入數據 CSV 檔案路徑')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='預測結果輸出目錄')
    parser.add_argument('--periods', type=int, default=30,
                       help='預測期數')
    parser.add_argument('--freq', type=str, default='D',
                       choices=['D', 'W', 'M'],
                       help='預測頻率（D=日，W=週，M=月）')
    parser.add_argument('--plot', action='store_true',
                       help='是否顯示視覺化圖表')
    
    args = parser.parse_args()
    
    # 載入數據
    print('載入數據...')
    df = load_data(args.data)
    print(f'載入了 {len(df)} 筆數據')
    
    # 初始化模型
    predictor = PowerPredictor()
    evaluator = ModelEvaluator()
    visualizer = PowerVisualizer()
    
    # 訓練模型
    print('訓練模型中...')
    predictor.train_models(df)
    print('模型訓練完成')
    
    # 進行預測
    print('進行預測...')
    start_date = df['date'].max().strftime('%Y-%m-%d')
    predictions = predictor.predict(start_date, args.periods, args.freq)
    
    # 儲存預測結果
    save_predictions(predictions, args.output)
    
    # 評估模型績效
    print('\n模型績效評估：')
    for model_name, pred_df in predictions.items():
        # 使用最後 N 筆數據進行評估
        actual = df.tail(len(pred_df))
        metrics = evaluator.calculate_metrics(
            actual['power_consumption'].values,
            pred_df['power_consumption'].values
        )
        print(f'\n{model_name} 模型：')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.2f}')
    
    # 視覺化（如果需要）
    if args.plot:
        print('\n繪製圖表...')
        visualizer.plot_consumption_trend(df)
        visualizer.plot_seasonal_patterns(df)
        evaluator.plot_predictions(df, predictions)

if __name__ == '__main__':
    main() 